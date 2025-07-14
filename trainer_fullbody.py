import argparse
import os
import numpy as np
import yaml
import random
import json 

#import trimesh 
import functools
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

from manip.data.hand_contact_data import HandContactDataset
from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from eval_metric import compute_metrics 

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

import blobfile as bf

from tqdm import tqdm

#model
from Long_CLIP.model_longclip import longclip
from semgeomo.data_loaders.humanml.scripts.motion_process import recover_from_ric
from semgeomo.data_loaders.humanml.common.skeleton import Skeleton
from semgeomo.diffusion.respace import SpacedDiffusion
from semgeomo.utils.fixseed import fixseed
from semgeomo.utils.parser_util import train_inpainting_args
from semgeomo.train.training_loop import TrainLoop
from semgeomo.data_loaders.get_data import get_dataset_loader
from semgeomo.utils.model_util import create_model_and_diffusion, load_pretrained_mdm_to_controlmdm
from semgeomo.humanml_utils import get_control_mask, HML_JOINT_NAMES
from semgeomo.diffusion.control_diffusion import ControlGaussianDiffusion
from semgeomo.model.ControlMDM import ControlMDM
from semgeomo.diffusion import logger
from semgeomo.utils import dist_util
from semgeomo.diffusion.fp16_util import MixedPrecisionTrainer
from semgeomo.diffusion.resample import LossAwareSampler, UniformSampler
from semgeomo.diffusion.resample import create_named_schedule_sampler
from semgeomo.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from semgeomo.utils.misc import load_model_wo_clip

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def kmeans_clustering(points, n_clusters):
    """
    Perform K-means clustering on the point cloud.
    
    Parameters:
    - points: np.ndarray of shape (N, 3) where N is the number of points.
    - n_clusters: int, the number of clusters to form.
    
    Returns:
    - labels: np.ndarray of shape (N,), the cluster label for each point.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto',random_state=0)
    kmeans.fit(points)
    labels = kmeans.labels_
    return labels

def cycle(dl):
    while True:
        for data in dl:
            yield data

class ControlDataLoader(object):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        mask_ratio = 1.0
        for motion, cond in super().__getattribute__('data').__iter__():
            n_joints = 22 if motion.shape[1] == 263 else 21
                
            unnormalized_motion = self.dataset.t2m_dataset.inv_transform(motion.permute(0, 2, 3, 1)).float()
            global_joints = recover_from_ric(unnormalized_motion, n_joints)
            global_joints = global_joints.view(-1, *global_joints.shape[2:]).permute(0, 2, 3, 1)
            global_joints.requires_grad = False
            cond['y']['global_joint'] = global_joints
            joint_mask = torch.tensor(get_control_mask(args.inpainting_mask, global_joints.shape, joint = ["left_wrist","right_wrist"], ratio = mask_ratio, dataset = args.dataset)).to(global_joints.device)
            #joint_mask = torch.tensor(get_control_mask(args.inpainting_mask, global_joints.shape, joint = ["left_foot","right_foot"], ratio = mask_ratio, dataset = args.dataset)).to(global_joints.device)
            #joint_mask = torch.tensor(get_control_mask(args.inpainting_mask, global_joints.shape, joint = "all", ratio = mask_ratio, dataset = args.dataset)).to(global_joints.device)
            joint_mask.requires_grad = False
            cond['y']['global_joint_mask'] = joint_mask
            yield motion, cond
    
    def __getattribute__(self, name):
        return super().__getattribute__('data').__getattribute__(name)
    
    def __len__(self):
        return len(super().__getattribute__('data'))
        
class Trainer(object):
    def __init__(
        self,
        opt,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=2000,###
        results_folder='./results',
        use_wandb=False,
        args,
        data
    ):
        super().__init__()

        self.use_wandb = use_wandb           

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 
        self.dataset_name = self.opt.dataset_name 

        self.data_root_folder = self.opt.data_root_folder 

        self.window = opt.window

        self.use_object_split = self.opt.use_object_split
        #self.prep_dataloader(window_size=opt.window)

        #self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.add_hand_processing = self.opt.add_hand_processing  

        self.for_quant_eval = self.opt.for_quant_eval 
        
        args.pretrained_path = self.opt.pretrained_path

        self.args = args
        

        #change model
        model, diffusion = create_model_and_diffusion(self.args,ModelClass=ControlMDM, DiffusionClass=ControlGaussianDiffusion)#######
        self.diffusion = diffusion
        self.ddp_model = model
        self.model = model

        self.cond_mode = model.cond_mode
        self.data = data
        self.model.mean = data.dataset.t2m_dataset.mean
        self.model.std = data.dataset.t2m_dataset.std
        self.diffusion.mean = data.dataset.t2m_dataset.mean
        self.diffusion.std = data.dataset.t2m_dataset.std
        self.ddp_model.mean = data.dataset.t2m_dataset.mean
        self.ddp_model.std = data.dataset.t2m_dataset.std

        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.use_posterior = args.use_posterior
        self.mask_ratio = args.mask_ratio

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        self.is_multi = hasattr(self.args, 'multi_arch')

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.schedule_sampler_type = 'uniform'  # 'control' if self.dataset in ['humanml'] else , control is not good
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())
        print(self.device)
        self.model.to(self.device)

        self.cur_mask_ratio = 1.0

        # train_platform_type = eval(args.train_platform_type)
        # train_platform = train_platform_type(args.save_dir)
        # train_platform.report_args(args, name='Args')
        # self.train_platform = train_platform

    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = HandContactDataset(train=True, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
            window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandContactDataset(train=False, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
            window=window_size, use_object_splits=self.use_object_split)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=0,drop_last = True))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=0,drop_last = True))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, pretrained_path=None):
        if pretrained_path != '':
            data = torch.load(pretrained_path)

            self.step = data['step']
            self.model.load_state_dict(data['model'], strict=False)
            self.ema.load_state_dict(data['ema'], strict=False)
            self.scaler.load_state_dict(data['scaler'])
        else:
            pass

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 
    

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Resuming model from checkpoint: {resume_checkpoint}...")
            state_dict = torch.load(resume_checkpoint, map_location='cpu')
            load_model_wo_clip(self.model, state_dict)
            self.model.to(dist_util.dev())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.batch_size):
            # Eliminates the microbatch feature
            assert i == 0
            micro = batch.to(dist_util.dev())
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.dataset_name,
                use_posterior = self.use_posterior,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

    def train(self):
        init_step = self.step 
        print("training begin============")
        mask_ratio = 1
        freeze_steps = 30000 if self.dataset_name in [ 'humanml',"behave","omomo","imhoi","interx","intergen","Unify"] else 15000 # kit is 15000
        mask_decay_steps = 60000 if self.dataset_name in [ 'humanml',"behave","omomo","imhoi","interx","intergen","Unify"] else 30000  # kit is 30000
        mask_final_step = 100000 if self.dataset_name in [ 'humanml',"behave","omomo","imhoi","interx","intergen","Unify"] else 50000  # kit is 50000
        print("Using mask ratio {}, decay from 1. mask_decay_steps={}, mask_final_step={}.".format(mask_ratio, mask_decay_steps, mask_final_step))
        print("model.output_process freeze at first {} steps, use posterior mean as bfgs update in training:{}".format(freeze_steps, self.use_posterior))
        print(self.num_epochs) 
        print(len(self.data))
        print(self.save_interval)
        print(self.args.pretrained_path)
        
        state_dict = torch.load(self.args.pretrained_path, map_location='cpu')
        load_pretrained_mdm_to_controlmdm(self.model, state_dict)
        
        self.model.to(self.device)
        
        for epoch in range(self.num_epochs):
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                
                self.run_step(motion, cond)   ####
                
                if self.step % self.log_interval == 0:
                    if self.step < mask_decay_steps:
                        cur_mask_ratio = 1.0
                        assert (self.cur_mask_ratio  - 1.0)<1e-4, "cur_mask_ratio should be 1.0 at the beginning"
                    elif self.step >= mask_decay_steps and self.step < mask_final_step:
                        cur_mask_ratio = 1 - (1 - self.mask_ratio) * (self.step - mask_decay_steps) / (mask_final_step - mask_decay_steps)
                    else:
                        cur_mask_ratio = self.mask_ratio
                    self.cur_mask_ratio = cur_mask_ratio

                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}], cur_mask_ratio[{:0.3f}]'.format(self.step+self.resume_step, v, cur_mask_ratio))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        # else:
                        #     self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                if self.step % self.save_interval == 0 and self.step > 0:
                    self.save()
                    if self.step >= freeze_steps:
                        self.model.unfreeze_block(self.model.output_process)
                    print(self.model.trainable_parameter_names())
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                self.step += 1
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        print('training complete')


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def run_train(opt, device,args):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    args = args

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                              short_db=args.short_db, cropping_sampler=args.cropping_sampler)
    data = ControlDataLoader(data)  ####
    
    trainer = Trainer(
        opt,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        args = args,
        data = data
    )

    #trainer.load(pretrained_path=opt.fullbody_checkpoint)

    trainer.train()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--fullbody_checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    
    parser.add_argument("--dataset_name", default=" ")
    parser.add_argument('--data_root_folder', default="", help='root folder for dataset')

    parser.add_argument('--datasettype',  default=None)
    parser.add_argument('--text',  default=False) #默认无文本

    parser.add_argument("--use_posterior", action="store_true")
    parser.add_argument('--save_dir', default='', help='save to this dir')
    parser.add_argument('--pretrained_path', default='', help='pretrained mdm ')



    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    args = train_inpainting_args()
    print(opt.dataset_name)
    args.dataset = opt.dataset_name
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    run_train(opt, device,args)
