import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from einops import rearrange

from manip.model.modules import CrossAttentionLayer, SelfAttentionBlock
from inspect import isfunction
from manip.model.cdm import ContactPerceiver
import torch
import clip
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

from manip.model.transformer_module import Decoder 

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


class CAffordJointDiffusionModel(nn.Module):
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
        contact_dim=2048,
        point_feat_dim=256,
        text_feat_dim=512,
        time_emb_dim=256
    ):
        super().__init__()
        
        # 定义常量
        num_heads = 8
        widening_factor = 1
        dropout = 0.1
        residual_dropout = 0.0
        encoder_self_attn_num_layers = 2

        # Transformer Diffusion Model 部分
        self.d_feats = d_feats
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k
        self.d_v = d_v
        self.max_timesteps = max_timesteps

        self.motion_transformer = Decoder(
            d_feats=d_input_feats, d_model=self.d_model,
            n_layers=self.n_dec_layers, n_head=self.n_head,
            d_k=self.d_k, d_v=self.d_v, max_timesteps=self.max_timesteps,
            use_full_attention=True
        )

        self.linear_out_transformer = nn.Linear(self.d_model, self.d_feats)

        # Afford Motion Diffusion Model 部分
        self.point_pos_emb = True

        self.language_adapter = nn.Linear(text_feat_dim, d_model, bias=True)

        self.encoder_adapter = nn.Linear(contact_dim + point_feat_dim + (3 if self.point_pos_emb else 0), 
                                         contact_dim, bias=True)
        self.decoder_adapter = nn.Linear(contact_dim, contact_dim, bias=True)

        self.encoder_cross_attn = CrossAttentionLayer(
            num_heads=num_heads,
            num_q_input_channels=d_model,
            num_kv_input_channels=contact_dim,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
        )

        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=encoder_self_attn_num_layers,
            num_heads=num_heads,
            num_channels=d_model,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
        )

        self.decoder_cross_attn = CrossAttentionLayer(
            num_heads=num_heads,
            num_q_input_channels=contact_dim,
            num_kv_input_channels=d_model,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
        )
        self.mul_attention = CrossAttentionLayer(
            num_heads=num_heads,
            num_q_input_channels=d_model,
            num_kv_input_channels=contact_dim,
            widening_factor=widening_factor,
            dropout=dropout,
            residual_dropout=residual_dropout,
        )

        # For noise level t embedding (shared across both parts)
        dim = 64
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, x, t, x_cond, padding_mask=None):
        time_embedding = self.time_mlp(t)[:, None, :]  # BS X 1 X d_model
        # Afford Motion Diffusion Model 输入的后 6 个通道
        x_afford = x[..., 6:]
        point_feat = x_cond[:, :, :259].detach()
        language_feat = x_cond[:, :, 259:(3 + 256 + self.d_model)].detach()

        x_afford = torch.cat([x_afford, point_feat], dim=-1)

        enc_kv = self.encoder_adapter(x_afford)
        language_feat = self.language_adapter(language_feat)
        enc_q = torch.cat([language_feat, time_embedding], dim=1)

        enc_q = self.encoder_cross_attn(enc_q, enc_kv).last_hidden_state
        enc_q = self.encoder_self_attn(enc_q).last_hidden_state

        dec_q = self.decoder_adapter(enc_kv)
        output_afford = self.decoder_cross_attn(dec_q, enc_q).last_hidden_state
        
        # Transformer Diffusion Model 输入的前 6 个通道
        src = x[..., :6]
        src_combined = torch.cat((src, x_cond), dim=-1)
        
        bs = src_combined.shape[0]
        num_steps = src_combined.shape[1] + 1
        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(x.device).bool()

        pos_vec = torch.arange(num_steps) + 1
        pos_vec = pos_vec[None, None, :].to(x.device).repeat(bs, 1, 1)

        data_input = src_combined.transpose(1, 2).detach()
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=time_embedding)
        # print(feat_pred.shape,output_afford.shape)
        mul_feat_pred = self.mul_attention(feat_pred[:, 1:], output_afford).last_hidden_state
        # print(mul_feat_pred.shape)
        output_transformer = self.linear_out_transformer(mul_feat_pred)
        
        # Concatenate Transformer and Afford outputs
        final_output = torch.cat((output_transformer, output_afford), dim=-1)
        
        return final_output

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
        text = False
    ):
        super().__init__()

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            )

        # Input: (BS*T) X 3 X N 
        # Output: (BS*T) X d X N, (BS*T) X d 
        # self.object_encoder = Pointnet() 

        obj_feats_dim = 256 
        text_feats_dim = 0
        if text:  
            text_feats_dim = 512
        d_input_feats = d_feats + 3 + obj_feats_dim + text_feats_dim
        contact_dim = 2048
        
        self.denoise_fn = CAffordJointDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps,contact_dim=contact_dim) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 
        self.cond_net = CondNet()

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

    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output = self.denoise_fn(x, t, x_cond, padding_mask)

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
    def p_sample(self, x, t, x_cond, padding_mask=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            padding_mask=padding_mask, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask)    

        return x # BS X T X D

    # @torch.no_grad()
    def sample(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None,text=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        self.bps_encoder.eval()

        # print(ori_x_cond.shape)
        #print(cond_mask)

        # (BPS representation) Encode object geometry to low dimensional vectors. 
        if text: 
            text_f = self.cond_net(text)
            B,T,c = ori_x_cond.shape
            text_f = text_f.unsqueeze(1).repeat(1,T,1)
            #print(text_f.shape)
            x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:]), text_f), dim=-1) # BS X T X (3+256+512)
        else:   
            x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1) # BS X T X (3+256) 


        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
       
        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond, padding_mask)
        # BS X T X D
            
        self.denoise_fn.train()
        self.bps_encoder.train()

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
        b, timesteps, d_input = x_start.shape # BS X T X D(3*4/3*2)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 

        model_out = self.denoise_fn(x, t, x_cond, padding_mask)

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

    def forward(self, x_start, ori_x_cond, cond_mask=None, padding_mask=None,text = None):
        # x_start: BS X T X D 
        # ori_x_cond: BS X T X D' (BPS representation)
        # ori_x_cond: BS X T X N X 3 (Point clouds)
        # actual_seq_len: BS 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        # (BPS representation) Encode object geometry to low dimensional vectors. 
        if text is not None: 
            text_f = self.cond_net(text)
            B,T,c = ori_x_cond.shape
            text_f = text_f.unsqueeze(1).repeat(1,T,1)
            x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:]), text_f), dim=-1) # BS X T X (3+256+512) 
        else:
            x_cond = torch.cat((ori_x_cond[:, :, :3], self.bps_encoder(ori_x_cond[:, :, 3:])), dim=-1) # BS X T X (3+256) 
        
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)
            x_cond = torch.cat((x_cond, x_pose_cond), dim=-1)
        curr_loss = self.p_losses(x_start, x_cond, t, padding_mask=padding_mask)

        return curr_loss
        
        
class CondNet(nn.Module):
    def __init__(self,text_encoder='clip'):
        super(CondNet, self).__init__()
        # self.device = device
        self.clip_model = self.load_and_freeze_clip()
        self.max_lang_len = 64
        # self.text_model_root_path = '/storage/group/4dvlab/congpsh/Diff-Motion/'
        self.text_encoder = text_encoder
        
    
    def load_and_freeze_clip(self):
        model_path = './pretrain/ViT-B-32.pt'
        clip_model = torch.load(model_path)
        # clip_model = load_state_dict(torch.load(model_path))
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 60  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()


    def _pad_utterance(self, token_id_seq, pad_val: int=0):
        if len(token_id_seq) > self.max_lang_len:
            return token_id_seq[0:self.max_lang_len]
        else:
            return token_id_seq + [pad_val] * (self.max_lang_len - len(token_id_seq))
        
    def forward(self,language):
        enc_text = self.encode_text(language)
        return enc_text
        