import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

from manip.model.transformer_module import Decoder 
import torch as th
import torch.optim as optim

import sys
sys.path.append("/storage/group/4dvlab/wangzy/intercontrol/data_loaders/humanml/scripts/")
#from motion_process import recover_from_ric

import joblib 


def de_normalize_jpos_min_max(normalized_jpos):
    min_max_mean_std_data_path = "/storage/group/4dvlab/wangzy/uni_regen/data_fps30/min_max_mean_std_data_window_100_cano_joints24.p"
    min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
    global_jpos_min = min_max_mean_std_jpos_data['global_jpos_min'].reshape(1,22, 3)
    global_jpos_max = min_max_mean_std_jpos_data['global_jpos_max'].reshape(1,22, 3)
    normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
    de_jpos = normalized_jpos * (global_jpos_max-global_jpos_min) + global_jpos_min
  
    return de_jpos # T X 22/24 X 3      
    
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=d_input_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, condition, padding_mask=None):
        # src: BS X T X D
        # noise_t: int 

        condition = condition.to(src.device)
        src = torch.cat((src, condition), dim=-1)
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1
        
        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=noise_t_embed)
        
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        batch_size=None,
    ):
        super().__init__()

        d_input_feats = 2*d_feats
            
        self.denoise_fn = TransformerDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        self.mse_loss = th.nn.MSELoss(reduction='mean')  ###

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def contact_joint_loss(self, pred_joint_a, pred_joint_b, mask,mask_obj):
        if mask.bool().sum() == 0:
            return th.tensor(0.0, device=pred_joint_a.device)

      
        desired_distance = mask.permute(0, 3, 1, 2).reshape(-1, 3)
        assert th.all(desired_distance[:, 0] == desired_distance[:, 1])
        assert th.all(desired_distance[:, 0] == desired_distance[:, 2])
        desired_distance = desired_distance[:, 0].contiguous()
        desired_distance = th.masked_select(desired_distance, desired_distance.bool())

        pred_joint_a = th.masked_select(pred_joint_a.permute(0, 3, 1, 2), mask[:, :, :, :].bool().permute(0, 3, 1, 2)).contiguous()#只挑出mask的点
        pred_joint_a = pred_joint_a.view(-1, 3)


        mask_obj_reshaped = mask_obj.permute(0, 3, 1, 2).reshape(-1, pred_joint_b.shape[1], 3)#nxvx3
        pred_joint_b = pred_joint_b.permute(0, 3, 1, 2).reshape(-1, pred_joint_b.shape[1], 3)
        pred_joint_b = th.masked_select(pred_joint_b, mask_obj_reshaped.bool()).view(-1, 3)#只挑出mask的点
        pred_joint_a = pred_joint_a.float()
        pred_joint_b = pred_joint_b.float()
        distance = th.cdist(pred_joint_a.unsqueeze(0), pred_joint_b.unsqueeze(0), p=2).mean(dim=-1) 
        loss = th.nn.functional.relu(distance - desired_distance)

        return loss.mean()
    
    def far_away_joint_loss(self, pred_joint_a, pred_joint_b, mask, mask_obj):

        if mask.bool().sum() == 0:
            return th.tensor(0.0, device=pred_joint_a.device)
    
        desired_distance = mask.permute(0, 3, 1, 2).reshape(-1, 3)
        assert th.all(desired_distance[:, 0] == desired_distance[:, 1])
        assert th.all(desired_distance[:, 0] == desired_distance[:, 2])
        desired_distance = desired_distance[:, 0].contiguous()
        desired_distance = th.masked_select(desired_distance, desired_distance.bool())

        pred_joint_a = th.masked_select(pred_joint_a.permute(0, 3, 1, 2), mask[:, :, :, :].bool().permute(0, 3, 1, 2)).contiguous()
        pred_joint_a = pred_joint_a.view(-1, 3)

        mask_obj_reshaped = mask_obj.permute(0, 3, 1, 2).reshape(-1, pred_joint_b.shape[1], 3)
        pred_joint_b = pred_joint_b.permute(0, 3, 1, 2).reshape(-1, pred_joint_b.shape[1], 3)
        pred_joint_b = th.masked_select(pred_joint_b, mask_obj_reshaped.bool()).view(-1, 3)
        pred_joint_a = pred_joint_a.float()
        pred_joint_b = pred_joint_b.float()
        distance = th.cdist(pred_joint_a.unsqueeze(0), pred_joint_b.unsqueeze(0), p=2).mean(dim=-1)
        loss = th.nn.functional.relu(distance - desired_distance)
        return loss.mean()
    

    def global_joint_bfgs_optimize(self,x,pointcloud,model_kwargs=None):

        loss = 0
        
        #loss for hand-position
        mean = model_kwargs["mean"]
        std = model_kwargs["std"]
        x = x.cpu().detach().numpy() * std + mean
        x = torch.tensor(x).float()
        x = recover_from_ric(x.reshape(-1,263),22)  #bs=1 263-level
        #x = de_normalize_jpos_min_max(x.detach().cpu().numpy().reshape(-1,22,3)) ##joint-level
        pred_joint = x.clone() #BS X T X 22 X 3
        pred_joint.requires_grad = True
        cond_joint = model_kwargs['global_joint']  #gt of the single_target_person
        mask = model_kwargs['global_joint_mask']
        bs,_,_,num_steps=mask.shape
        pred_joint = pred_joint.reshape((bs,22,3,num_steps)).to(mask.device)
        cond_joint = cond_joint.reshape((bs,22,3,num_steps)).to(mask.device)
        pred_joint = th.masked_select(pred_joint, mask.bool())
        cond_joint = th.masked_select(cond_joint, mask.bool())
        assert pred_joint.shape == cond_joint.shape, f"pred_joint: {pred_joint.shape}, cond_joint: {cond_joint.shape}"
        loss += self.mse_loss(pred_joint, cond_joint)

        # loss for contact
        '''interact = pointcloud.reshape((bs,1024,3,num_steps)) #BS X V X 3 X T
        contact_mask = model_kwargs['contact_mask'] #BS X pairs x 22 X 3 X T
        far_away_mask = model_kwargs['far_away_mask']
        contact_mask_obj = model_kwargs['object_contact_mask'] #BS X pairs x V X 3 X T
        far_away_mask_obj = model_kwargs['object_far_mask']
        p = contact_mask.shape[1]
        target = x.clone()
        target = target.reshape((bs,22,3,num_steps)).to(contact_mask.device)  #BS X 22 X 3 X T
        for b in range(bs):
          loss += self.contact_joint_loss(target[b].unsqueeze(0).repeat(p, 1, 1, 1), interact[b].unsqueeze(0).repeat(p, 1, 1, 1), contact_mask[b],contact_mask_obj[b])
          loss += self.far_away_joint_loss(target[b].unsqueeze(0).repeat(p, 1, 1, 1), interact[b].unsqueeze(0).repeat(p, 1, 1, 1), far_away_mask[b],far_away_mask_obj[b])'''
        
        return loss
    
    def condition_mean_bfgs(self, x_mean, pointcloud,num_condition, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        with th.enable_grad():
            x_mean = x_mean.clone().detach().contiguous().requires_grad_(True)
            pointcloud = pointcloud
            def closure():
                lbfgs.zero_grad()
                objective = self.global_joint_bfgs_optimize(x_mean,pointcloud,model_kwargs)
                objective.backward()
                return objective
            lbfgs = optim.LBFGS([x_mean],
                    history_size=10, 
                    max_iter=4, 
                    line_search_fn="strong_wolfe")
            for _ in range(num_condition):
                lbfgs.step(closure)
        #loss_value = self.global_joint_bfgs_optimize(x_mean, model_kwargs).item()
        return x_mean  #, loss_value
    
    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised,pointcloud,model_kwargs):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output = self.denoise_fn(x, t, x_cond, padding_mask)

        #loss-guidance
        '''num_condition = 5
        pointcloud = pointcloud
        model_output = self.condition_mean_bfgs(model_output, pointcloud,num_condition,model_kwargs=model_kwargs)  # , loss_value'''
        
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, padding_mask=None, clip_denoised=True,pointcloud=None,model_kwargs=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            padding_mask=padding_mask, clip_denoised=clip_denoised,pointcloud = pointcloud,model_kwargs=model_kwargs)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, padding_mask=None, pointcloud=None, model_kwargs=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask,pointcloud=pointcloud,model_kwargs=model_kwargs)    

        return x # BS X T X D

    @torch.no_grad()
    def sample(self, x_start, cond_mask=None, padding_mask=None, pointcloud=None,model_kwargs=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)
       
            x_cond = x_pose_cond 

        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond, padding_mask,pointcloud,model_kwargs) 
        # BS X T X D
            
        self.denoise_fn.train()

        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, x_cond, t, noise=None, padding_mask=None):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T 
        b, timesteps, d_input = x_start.shape # BS X T X D(3+n_joints*4)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
        
        #print("aaa")
            
        model_out = self.denoise_fn(x, t, x_cond, padding_mask)
        
        #print("hhh")

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

    def forward(self, x_start, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D 
        # ori_x_cond: BS X T X D' 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)

            x_cond = x_pose_cond 
        
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss
        