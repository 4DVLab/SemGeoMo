import torch as th
import torch.optim as optim
import numpy as np
import math
import sys
import torch
import torch.nn.functional as F
from .respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor, ModelMeanType, ModelVarType, LossType
import sys
sys.path.append("..")
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def euclidean_distance(x, y, z, x1, y1, z1):
    return math.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)
    
def z_score_normalize(data):
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

def z_score_normalize_torch(data):  
   mean = data.mean(dim=(0, 1), keepdim=True)  
   std = data.std(dim=(0, 1), keepdim=True)  
   normalized_data = (data - mean) / std  
   return normalized_data

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class ControlGaussianDiffusion(SpacedDiffusion):

    def inv_transform(self, data):
        assert self.std is not None and self.mean is not None
        #assert data.requires_grad == True
        std = th.tensor(self.std, dtype=data.dtype, device=data.device, requires_grad=False)
        mean = th.tensor(self.mean, dtype=data.dtype, device=data.device, requires_grad=False)
        output = th.add(th.mul(data, std), mean)
        return output
    
    def q_sample(self, x_start, t, noise=None, model_kwargs=None):
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        noise *= 1. #- model_kwargs['y']['inpainting_mask']

        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            )
    
    def feet_floor_contact(self, pred_x): 
    
       pred_x = self.humanml_to_global_joint(pred_x).to("cuda:0") 
       pred_x = pred_x.reshape(-1, 22, 3).to("cuda:0")  # T x 22 x 3
       human_jnts = pred_x
      
       left_toe_idx = 10
       right_toe_idx = 11
      
       l_toe_height = human_jnts[:, left_toe_idx, 1]  #+y
       r_toe_height = human_jnts[:, right_toe_idx, 1] 
       support_foot_height = torch.minimum(l_toe_height, r_toe_height)
       target_height = torch.ones_like(support_foot_height) * 0.02
      
       loss_feet_floor_contact = F.mse_loss(support_foot_height, target_height)
  
       return loss_feet_floor_contact

    
    def feet_velocity_loss(self, pred_x):
        pred_x = self.humanml_to_global_joint(pred_x).to("cuda:0") 
        pred_x = pred_x.reshape(-1, 22, 3).to("cuda:0")  # T x 22 x 3
        human_jnts = pred_x
    
        left_toe_idx = 10
        right_toe_idx = 11
    
        l_toe_pos = human_jnts[:, left_toe_idx, :]
        r_toe_pos = human_jnts[:, right_toe_idx, :]
        
        
        l_toe_vel = l_toe_pos[1:] - l_toe_pos[:-1]
        r_toe_vel = r_toe_pos[1:] - r_toe_pos[:-1]
    
        
        l_toe_height = l_toe_pos[:, 1]
        r_toe_height = r_toe_pos[:, 1]
    
        foot_contact_mask = (l_toe_height < 0.02) | (r_toe_height < 0.02) 
  
        l_toe_vel_loss = l_toe_vel[foot_contact_mask[1:]].norm(dim=-1).mean()
        r_toe_vel_loss = r_toe_vel[foot_contact_mask[1:]].norm(dim=-1).mean()
    
        return (l_toe_vel_loss + r_toe_vel_loss) / 2
        
    def feet_acceleration_loss(self, pred_x):
        pred_x = self.humanml_to_global_joint(pred_x).to("cuda:0") 
        pred_x = pred_x.reshape(-1, 22, 3).to("cuda:0")  # T x 22 x 3
        human_jnts = pred_x
    
        left_toe_idx = 10
        right_toe_idx = 11
    
        l_toe_pos = human_jnts[:, left_toe_idx, :]
        r_toe_pos = human_jnts[:, right_toe_idx, :]
        
       
        l_toe_vel = l_toe_pos[1:] - l_toe_pos[:-1]
        r_toe_vel = r_toe_pos[1:] - r_toe_pos[:-1]
    
        l_toe_acc = l_toe_vel[1:] - l_toe_vel[:-1]
        r_toe_acc = r_toe_vel[1:] - r_toe_vel[:-1]
    
        
        l_toe_height = l_toe_pos[:, 1]
        r_toe_height = r_toe_pos[:, 1]
    
        foot_contact_mask = (l_toe_height < 0.02) | (r_toe_height < 0.02)
    
        
        l_toe_acc_loss = l_toe_acc[foot_contact_mask[2:]].norm(dim=-1).mean()
        r_toe_acc_loss = r_toe_acc[foot_contact_mask[2:]].norm(dim=-1).mean()
    
        return (l_toe_acc_loss + r_toe_acc_loss) / 2
        
    def feet_contact_stability_loss(self, pred_x):
        pred_x = self.humanml_to_global_joint(pred_x).to("cuda:0") 
        pred_x = pred_x.reshape(-1, 22, 3).to("cuda:0")  # T x 22 x 3
        human_jnts = pred_x
    
        left_toe_idx = 10
        right_toe_idx = 11
    
        l_toe_height = human_jnts[:, left_toe_idx, 1]
        r_toe_height = human_jnts[:, right_toe_idx, 1]
    
        foot_contact_mask = (l_toe_height < 0.02) | (r_toe_height < 0.02)
    
        l_toe_height_change = l_toe_height[1:] - l_toe_height[:-1]
        r_toe_height_change = r_toe_height[1:] - r_toe_height[:-1]
    
        l_toe_stability_loss = l_toe_height_change[foot_contact_mask[1:]].abs().mean()
        r_toe_stability_loss = r_toe_height_change[foot_contact_mask[1:]].abs().mean()
    
        return (l_toe_stability_loss + r_toe_stability_loss) / 2


    def global_joint_bfgs_optimize(self,x,person2,pointcloud,gt,aff,model_kwargs=None):
        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        
        #joint-loss
        pred_joint = self.humanml_to_global_joint(x).to("cuda:0") 
        cond_joint = model_kwargs['y']['global_joint'] #pred joint of stage1
        mask = model_kwargs['y']['global_joint_mask']
        pred_joint = th.masked_select(pred_joint, mask.bool())
        cond_joint = th.masked_select(cond_joint, mask.bool())
        #print(pred_joint.requires_grad) #True
        #print(cond_joint.requires_grad)  #False
        assert pred_joint.shape == cond_joint.shape, f"pred_joint: {pred_joint.shape}, cond_joint: {cond_joint.shape}"
        joint_loss = self.mse_loss(pred_joint, cond_joint)
        
        loss = joint_loss
        #print(joint_loss)

        return loss
    
    def soft_loss_bfgs_optimize(self,x,person2,pointcloud,gt,aff,model_kwargs=None): #for affordmotion
        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        
        #soft-loss
        pred_joint = self.humanml_to_global_joint(x).to("cuda:0") 
        cond_joint = model_kwargs['y']['global_joint'] #gt_hand_jont
        pred_joint = pred_joint.permute(0,3,1,2)[:,:,20:,:]  
        cond_joint = cond_joint.permute(0,3,1,2)[:,:,20:,:] 
        pointcloud = pointcloud.to(pred_joint.device)
        # print(pred_joint.shape)
        # print(cond_joint.shape)
        # print(pointcloud.shape)
        b, t, j, d = pred_joint.shape  
        _, _, o, _ = pointcloud.shape  

        actual_flat = pred_joint.reshape(-1, j, d)  # (b * t, j, d)
        expected_flat = cond_joint.reshape(-1, j, d)  # (b * t, j, d)

        predicted_distances = torch.norm(actual_flat.unsqueeze(2) - pointcloud.reshape(-1, 1, o, d), dim=-1)  # (b * t, j, o)
        true_distances = torch.norm(expected_flat.unsqueeze(2) - pointcloud.reshape(-1, 1, o, d), dim=-1)  # (b * t, j, o)
        
        distance_diff = torch.abs(predicted_distances - true_distances)  # (b * t, j, o)
        
        # For the person, compute the minimum distance difference across all object points
        min_distance_diff = torch.min(distance_diff, dim=-1)[0]  # (b * t, j) 
        
        # Average the losses across two hand joints
        loss = torch.mean(min_distance_diff, dim=-1).reshape(b,t)  # (b  t)

        assert not torch.isnan(loss).any(), "Loss contains NaN values"

        #mean loss over allframes
        loss = torch.mean(loss,dim=1)

        if b >1:
            loss = torch.mean(loss,dim=0)  #over batch
        
        return loss
    
    def humanml_to_global_joint(self, x):
        n_joints = 22 if x.shape[1] == 263 else 21
        pred_joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
        assert pred_joint.shape[1] == 1
        pred_joint = recover_from_ric(pred_joint, n_joints)
        pred_joint = pred_joint.view(-1, *pred_joint.shape[2:]).permute(0, 2, 3, 1)
        return pred_joint
    
    def global_joint_position_conditioning(self, x, model_kwargs=None):
        n_joints = 22 if x.shape[1] == 263 else 21
        assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
        pred_joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
        pred_joint = recover_from_ric(pred_joint, n_joints)
        pred_joint = pred_joint.view(-1, *pred_joint.shape[2:]).permute(0, 2, 3, 1)
        #pred_joint.requires_grad = True
        assert pred_joint.shape == model_kwargs['y']['global_joint'].shape == model_kwargs['y']['global_joint_mask'].shape, f"pred_joint: {pred_joint.shape}, global_joint: {model_kwargs['y']['global_joint'].shape}, global_joint_mask: {model_kwargs['y']['global_joint_mask'].shape}"
        loss = self.global_joint_condition_loss(pred_joint, model_kwargs['y']['global_joint'], model_kwargs['y']['global_joint_mask'])
        diff_scale = ((pred_joint.clamp(min=1e-4) - model_kwargs['y']['global_joint'].clamp(min=1e-4)).abs() / model_kwargs['y']['global_joint'].clamp(min=1e-4).abs()).mean().item()
        #loss.requires_grad = True
        gradient = th.autograd.grad(loss, x,            
            grad_outputs=th.ones_like(loss),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradient.clone().detach(), loss.item(), diff_scale

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        use_posterior=False,
        person2 = None, ##!
        pointcloud = None, ##!
        gt = None,
        aff =None,
    ):
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        #assert use_posterior == False
        p_mean_variance_func = self.p_mean_variance_bfgs_posterior if use_posterior else self.p_mean_variance_bfgs_x0
        out = p_mean_variance_func(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            k_first = self.bfgs_times_first,
            k_last = self.bfgs_times_last,
            person2 = person2,
            pointcloud = pointcloud,
            gt = gt,
            aff = aff,
        )
        
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def condition_mean_with_grad(self, cond_fn, x_mean, x_var, t, strength, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        with th.enable_grad():
            x_mean = x_mean.clone().detach().requires_grad_(True)
            gradient, loss_value, diff_scale = cond_fn(x_mean, model_kwargs)  #  p_mean_var["mean"]
            gradient_guidance = - strength * gradient.float()  # x_var.clamp(min = 0.01) 
            new_mean = (x_mean + gradient_guidance).clone().detach()
        return new_mean, loss_value, gradient_guidance.clone().detach().abs().cpu(), x_mean.clone().detach().abs().cpu(), diff_scale

        
    def condition_mean_bfgs(self, x_mean, num_condition, person2=None, pointcloud=None, gt=None, aff=None, model_kwargs=None, optimize_foot=False):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to x.
    
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        
        with th.enable_grad():
            x_mean = x_mean.clone().detach().contiguous().requires_grad_(True)
            
            person2 = person2
            pointcloud = pointcloud ##!
            gt = gt
            aff = aff
            optimize_foot = optimize_foot
            adam_optimizer = optim.Adam([x_mean], lr=1e-3)
    
            def lbfgs_closure():  #joint loss / soft_loss
                 lbfgs_optimizer.zero_grad()
                 objective = self.global_joint_bfgs_optimize(x_mean, person2, pointcloud, gt, aff, model_kwargs)  #only hand 
                 objective.backward()  
                 return objective
    
            lbfgs_optimizer = optim.LBFGS([x_mean], history_size=10, max_iter=4, line_search_fn="strong_wolfe")
    
            for _ in range(num_condition):  #foot loss
                 lbfgs_optimizer.step(lbfgs_closure)
            
            if optimize_foot:
              adam_optimizer.zero_grad()  
                
              feet_contact_loss = self.feet_floor_contact(x_mean)
              feet_vel_loss = self.feet_velocity_loss(x_mean)
              feet_acc_loss = self.feet_acceleration_loss(x_mean)
              feet_stability_loss = self.feet_contact_stability_loss(x_mean)
              foot_loss = feet_contact_loss + 0.5 * feet_vel_loss + 0.5 * feet_acc_loss + 0.5 * feet_stability_loss

            foot_loss.backward()  
            adam_optimizer.step()
    
        return x_mean  


    def p_mean_variance_bfgs_x0(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, 
        k_first = 1,
        k_last = 10,
        t_threshold = 10,
        person2 = None,
        pointcloud = None,
        gt = None,
        aff =None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        original_model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output = original_model_output.clone().detach()

        self.model_var_type = ModelVarType(self.model_var_type.value)
        self.model_mean_type = ModelMeanType(self.model_mean_type.value)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)


        # loss-guided condition!
        #assert k_first ==1, "k_first must be 1, {}".format(k_first)
        num_condition = k_first if t[0] >= t_threshold else k_last  # t[0] count from 1000 to 1, assume all t are equal
        model_output = self.condition_mean_bfgs(model_output, num_condition, person2 = person2, pointcloud = pointcloud,gt = gt ,aff=aff,model_kwargs=model_kwargs,optimize_foot=True)  # , loss_value

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def p_mean_variance_bfgs_posterior(  #####we use
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, 
        k_first = 1,
        k_last = 10,
        t_threshold = 10,
        person2 = None,
        pointcloud = None, 
        gt = None,
        aff = None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        original_model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        model_output = original_model_output.clone().detach()

        # for k in {
        #     ModelVarType.FIXED_LARGE: 1,
        #     ModelVarType.FIXED_SMALL: 2,
        # }.keys():
        #     print(f"Key: {k}, Same as self.model_var_type? {k is self.model_var_type}")

        self.model_var_type = ModelVarType(self.model_var_type.value)
        self.model_mean_type = ModelMeanType(self.model_mean_type.value)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)


        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                # print('clip_denoised', clip_denoised)
                return x.clamp(-1, 1)
            return x
        

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:  #we use
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        # loss-guided condition
        #assert k_first ==1, "k_first must be 1, {}".format(k_first)
        #num_condition = k_first if t[0] >= t_threshold else k_last  # t[0] count from 1000 to 1, assume all t are equal
        num_condition = 5  #ours:5
        model_mean = self.condition_mean_bfgs(model_mean, num_condition, person2 = person2 , pointcloud = pointcloud,gt=gt,aff=aff,model_kwargs=model_kwargs,optimize_foot=True)  # , loss_value
        

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None,
                        use_posterior = True,
                        k_first = 1,
                        k_last = 10,
                        t_threshold = 10,):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        # enc = model.model._modules['module']
        model = self._wrap_model(model)
         
        enc = model.model
        mask = model_kwargs['y']['mask']
        # get_xyz = lambda sample: enc.rot2xyz(sample, mask=None, pose_rep=enc.pose_rep, translation=enc.translation,
        #                                      glob=enc.glob,
        #                                      # jointstype='vertices',  # 3.4 iter/sec # USED ALSO IN MotionCLIP
        #                                      jointstype='smpl',  # 3.4 iter/sec
        #                                      vertstrans=False,
        #                                      name = None)

        def get_xyz(x):
            joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
            assert joint.shape[1] == 1
            joint = recover_from_ric(joint, 22)
            joint = joint.view(-1, *joint.shape[2:]).permute(0, 2, 3, 1)
            return joint
        
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise, model_kwargs=model_kwargs)  #add_noise
        
        #assert k_first == 1, "k_first must be 1, {}".format(k_first)
        #assert k_last == 10, "k_last must be 10, {}".format(k_last)
        #For train on xo, use_posterior=false
        #assert use_posterior == True, "use_posterior must be True, {}".format(use_posterior)
        if use_posterior:
            '''
            # loss-guided condition in training time
            if t[0] >= t_threshold:
                assert (t >= t_threshold).all(), f"all t should be >=10 or <10 : t={t}"
                num_condition = k_first # else k_last
            else:
                num_condition = k_last
                assert (t < t_threshold).all(), f"all t should be >=10 or <10 : t={t}"
            '''
            person2 = None
            pointcloud = None 
            num_condition = k_first
            self.model_mean_type = ModelMeanType(self.model_mean_type.value)
            self.loss_type = LossType(self.loss_type.value)
            x_t = self.condition_mean_bfgs(x_t, num_condition, person2 = person2,pointcloud = pointcloud, model_kwargs=model_kwargs,optimize_foot=True)

        terms = {}
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape, "model_output {},  target {}, x_start {}".format(model_output.shape ,target.shape ,x_start.shape)  # [bs, njoints, nfeats, nframes]

            terms["rot_mse"] = self.masked_l2(target, model_output, mask) # mean_flat(rot_mse)

            terms["loss"] = terms["rot_mse"]

                          
        else:
            raise NotImplementedError(self.loss_type)

        return terms

