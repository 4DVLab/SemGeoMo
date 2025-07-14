import argparse
import os
import numpy as np
import yaml
import random
import json 
import torch as th

import trimesh 

from tqdm import tqdm
from pathlib import Path
import pickle
import imageio
import spacy
import itertools
import chardet
import rich
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 
from plyfile import PlyData, PlyElement
from pywavefront import Wavefront
from ema_pytorch import EMA
from multiprocessing import cpu_count
import chardet
import codecs as cs

import trimesh 
import time 
from os.path import join as pjoin

from sklearn.cluster import KMeans

from manip.data.hand_contact_data import HandContactDataset

from manip.model.transformer_hand_foot_manip_cond_diffusion_model import CondGaussianDiffusion 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from eval_metric import compute_metrics, compute_s1_metrics,compute_collision

from matplotlib import pyplot as plt

import logging
import chardet
from tqdm import tqdm
import codecs as cs
import random
import shutil

from bps_torch.bps import bps_torch

from tqdm import tqdm

#model
from semgeomo.data_loaders.humanml.scripts.motion_process import recover_from_ric
from semgeomo.data_loaders.humanml.common.skeleton import Skeleton
from semgeomo.diffusion.respace import SpacedDiffusion
from semgeomo.utils.fixseed import fixseed
from semgeomo.utils.parser_util import edit_control_args
from semgeomo.train.training_loop import TrainLoop
from semgeomo.data_loaders.get_data import get_dataset_loader
from semgeomo.utils.model_util import create_model_and_diffusion, load_pretrained_mdm_to_controlmdm
from semgeomo.humanml_utils import get_control_mask, HML_JOINT_NAMES
from semgeomo.diffusion.control_diffusion import ControlGaussianDiffusion
from semgeomo.model.ControlMDM import ControlMDM #####
from semgeomo.diffusion import logger
from semgeomo.utils import dist_util
from semgeomo.diffusion.fp16_util import MixedPrecisionTrainer
from semgeomo.diffusion.resample import LossAwareSampler, UniformSampler
from semgeomo.diffusion.resample import create_named_schedule_sampler
from semgeomo.data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from semgeomo.data_loaders.get_data import get_dataset_loader
from semgeomo.utils.misc import load_model_wo_clip
from semgeomo.utils.model_util import load_controlmdm_and_diffusion
from semgeomo.model.cfg_sampler import wrap_model
import semgeomo.data_loaders.humanml.utils.paramUtil as paramUtil
from semgeomo.data_loaders.humanml.utils.plot_script import plot_3d_motion

def kmeans_clustering(points, n_clusters):
    """
    Perform K-means clustering on the point cloud.
    
    Parameters:
    - points: np.ndarray of shape (N, 3) where N is the number of points.
    - n_clusters: int, the number of clusters to form.
    
    Returns:
    - labels: np.ndarray of shape (N,), the cluster label for each point.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10,random_state=0)
    kmeans.fit(points)
    labels = kmeans.labels_
    return labels


def cycle(dl):  
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=10000,###
        results_folder='./results',
        use_wandb=False,  
        args,
        data,
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = 1  #every time sample 1 
        self.guidance_param = 2.5
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.bps_path = "./manip/data/bps.pt"
        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()
        self.obj_bps = self.bps['obj'] 

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        self.window = opt.window

        self.use_object_split = self.opt.use_object_split 

        self.data_root_folder = self.opt.data_root_folder 
        self.dataset_name = self.opt.dataset_name 
        #print(self.data_root_folder)

        self.prep_dataloader(window_size=opt.window)

        # self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.add_hand_processing = self.opt.add_hand_processing 

        self.for_quant_eval = self.opt.for_quant_eval 

        self.use_gt_hand_for_eval = self.opt.use_gt_hand_for_eval 
        
        self.args = args
        self.data = data
        DiffusionClass = ControlGaussianDiffusion 
        #change model
        model, diffusion = load_controlmdm_and_diffusion(self.args, data, dist_util.dev(), ModelClass=ControlMDM, DiffusionClass=DiffusionClass)###
        self.diffusion = diffusion
        self.model = model
        self.model = self.model.to("cuda:0")
        self.use_posterior = args.use_posterior
        
        self.model.mean = data.dataset.t2m_dataset.mean
        self.model.std = data.dataset.t2m_dataset.std
        self.diffusion.mean = data.dataset.t2m_dataset.mean
        self.diffusion.std = data.dataset.t2m_dataset.std


    def prep_dataloader(self, window_size):
        # Define dataset
        # train_dataset = HandContactDataset(train=True, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
        #     window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandContactDataset(train=False, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
            window=window_size, use_object_splits=self.use_object_split)
        # self.ds = train_dataset 
        # self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
        #     shuffle=True, pin_memory=True, num_workers=0,drop_last = True))
        self.val_ds = val_dataset
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=0,drop_last=True)) #
    
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
            
    def load_aff(self, pretrained_path=None):
        if pretrained_path != '':
            data = torch.load(pretrained_path)

            self.step = data['step']
            self.model.load_state_dict(data['model'], strict=False)
            self.ema_aff.load_state_dict(data['ema'], strict=False)
            self.scaler.load_state_dict(data['scaler'])
        else:
            pass



    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: 4(2) X 3  
        lhand_color = np.asarray([255, 87, 51])  # red 
        rhand_color = np.asarray([17, 99, 226]) # blue
        lfoot_color = np.asarray([134, 17, 226]) # purple 
        rfoot_color = np.asarray([22, 173, 100]) # green 

        color_list = [lhand_color, rhand_color, lfoot_color, rfoot_color]

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])
            
            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=color_list[idx],
                process=False)

            result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
            output_file = open(ball_mesh_path.replace(".ply", "_"+str(idx)+".ply"), "wb+")
            output_file.write(result)
            output_file.close()

    def export_to_mesh(self, mesh_verts, mesh_faces, mesh_path):
        dest_mesh = trimesh.Trimesh(
            vertices=mesh_verts,
            faces=mesh_faces,
            process=False)

        result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
        output_file = open(mesh_path, "wb+")
        output_file.write(result)
        output_file.close()

    def process_hand_foot_contact_jpos(self, hand_foot_jpos, object_mesh_verts, object_mesh_faces, obj_rot):
        # hand_foot_jpos: T X 2 X 3 
        # object_mesh_verts: T X Nv X 3 
        # object_mesh_faces: Nf X 3 
        # obj_rot: T X 3 X 3 
        all_contact_labels = []
        all_object_c_idx_list = []
        all_dist = []

        obj_rot = torch.from_numpy(obj_rot).to(hand_foot_jpos.device).float()
        object_mesh_verts = object_mesh_verts.to(hand_foot_jpos.device)

        num_joints = hand_foot_jpos.shape[1]
        num_steps = hand_foot_jpos.shape[0]

        threshold = 0.03 # Use palm position, should be smaller. 
       
        joint2object_dist = torch.cdist(hand_foot_jpos, object_mesh_verts.to(hand_foot_jpos.device)) # T X 2 X Nv 
     
        all_dist, all_object_c_idx_list = joint2object_dist.min(dim=2) # T X 2
        all_contact_labels = all_dist < threshold # T X 2

        new_hand_foot_jpos = hand_foot_jpos.clone() # T X 2 X 3 

        # For each joint, scan the sequence, if contact is true, then use the corresponding object idx for the 
        # rest of subsequence in contact. 
        for j_idx in range(num_joints):
            continue_prev_contact = False 
            for t_idx in range(num_steps):
                if continue_prev_contact:
                    relative_rot_mat = torch.matmul(obj_rot[t_idx], reference_obj_rot.inverse())
                    curr_contact_normal = torch.matmul(relative_rot_mat, contact_normal[:, None]).squeeze(-1)

                    new_hand_foot_jpos[t_idx, j_idx] = object_mesh_verts[t_idx, subseq_contact_v_id] + \
                        curr_contact_normal  # 3  
                
                elif all_contact_labels[t_idx, j_idx] and not continue_prev_contact: # The first contact frame 
                    subseq_contact_v_id = all_object_c_idx_list[t_idx, j_idx]
                    subseq_contact_pos = object_mesh_verts[t_idx, subseq_contact_v_id] # 3 

                    contact_normal = new_hand_foot_jpos[t_idx, j_idx] - subseq_contact_pos # Keep using this in the following frames. 

                    reference_obj_rot = obj_rot[t_idx] # 3 X 3 

                    continue_prev_contact = True 

        return new_hand_foot_jpos 

    def gen_vis_res(self, all_res_list, data_dict, step, vis_gt=False, vis_tag=None):
        # all_res_list: BS X T X 12  
        lhand_color = np.asarray([255, 87, 51])  # red 
        rhand_color = np.asarray([17, 99, 226]) # blue
        lfoot_color = np.asarray([134, 17, 226]) # purple 
        rfoot_color = np.asarray([22, 173, 100]) # green 

        contact_pcs_colors = []
        contact_pcs_colors.append(lhand_color)
        contact_pcs_colors.append(rhand_color)
        contact_pcs_colors.append(lfoot_color)
        contact_pcs_colors.append(rfoot_color)
        contact_pcs_colors = np.asarray(contact_pcs_colors) # 4 X 3 
        
        seq_names = data_dict['seq_name'] #
        seq_len = data_dict['seq_len'].detach().cpu().numpy() 

        # obj_rot = data_dict['obj_rot_mat'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 X 3
        obj_com_pos = data_dict['obj_com_pos'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 

        num_seq, num_steps, _ = all_res_list.shape
        
        normalized_gt_hand_foot_pos = data_dict['gt_hands']#.reshape(-1, num_steps, 2, 3) 
        
        #pred_hand = all_res_list[:,:,:2*3]
        
        # Denormalize hand only 
        pred_hand_foot_pos = self.val_ds.de_normalize_jpos_min_max_hand_foot(all_res_list.cpu(), hand_only=True)
        gt_hand_foot_pos = self.val_ds.de_normalize_jpos_min_max_hand_foot(normalized_gt_hand_foot_pos.cpu(),hand_only=True) # BS X T X 2 X 3
        all_processed_hand_jpos = pred_hand_foot_pos.clone() 
        
        for seq_idx in range(num_seq):
            object_name = seq_names[seq_idx].split("_")[1]
            obj_scale = data_dict['obj_scale'][seq_idx].detach().cpu().numpy()
            obj_trans = data_dict['obj_trans'][seq_idx].detach().cpu().numpy()
            obj_rot = data_dict['obj_rot_mat'][seq_idx].detach().cpu().numpy()

            obj_bottom_scale = None 
            obj_bottom_trans = None 
            obj_bottom_rot = None 

            obj_mesh_verts, obj_mesh_faces = self.val_ds.load_object_geometry(object_name, \
            obj_scale, obj_trans, obj_rot, \
            obj_bottom_scale, obj_bottom_trans, obj_bottom_rot)

            # Add postprocessing for hand positions. 
            if self.add_hand_processing:
                curr_seq_pred_hand_foot_jpos = self.process_hand_foot_contact_jpos(pred_hand_foot_pos[seq_idx], \
                                    obj_mesh_verts, obj_mesh_faces, obj_rot)

                all_processed_hand_jpos[seq_idx] = curr_seq_pred_hand_foot_jpos 
            else:
                curr_seq_pred_hand_foot_jpos = pred_hand_foot_pos[seq_idx]

        if self.use_gt_hand_for_eval:
            all_processed_hand_jpos = self.val_ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.cuda())
        else:
            all_processed_hand_jpos = self.val_ds.normalize_jpos_min_max_hand_foot(all_processed_hand_jpos) # BS X T X 2 X 3 

        gt_hand_foot_pos = self.val_ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos)

        return all_processed_hand_jpos, gt_hand_foot_pos  
    

    # def gen_vis_res_fullbody(self, all_res_list, data_dict, step, vis_gt=False, vis_tag=None):
    #     # all_res_list: bs X T X 22 x 3
    #     num_seq = all_res_list.shape[0]  #bs:1

    #     num_joints = 22
    
    #     global_jpos = all_res_list.reshape(num_seq, -1, num_joints, 3) # bs X T X 22 X 3 
    #     global_root_jpos = global_jpos[:, :, 0, :].clone() # bs X T X 3 

    
    #     # Used for quantitative evaluation. 
    #     human_verts_list = []

    #     obj_verts_list = []
    #     obj_faces_list = [] 

    #     actual_len_list = []

    #     for idx in range(num_seq): #in batch_size
    #         #print(idx)
    #         curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
    #         curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
    #         curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
    #         curr_global_root_jpos = global_root_jpos[idx] # T X 3

    #         root_trans = curr_global_root_jpos #+ curr_trans2joint # T X 3 
         
    #         # Generate global joint position 
    #         betas = data_dict['betas'][idx]
    #         gender = data_dict['gender'][idx]
    #         curr_obj_rot_mat = data_dict['obj_rot_mat'][idx]
    #         curr_obj_trans = data_dict['obj_trans'][idx]
    #         curr_obj_scale = data_dict['obj_scale'][idx]
    #         curr_seq_name = data_dict['seq_name'][idx]
    #         object_name = curr_seq_name.split("_")[1]
        

    #         # Get human verts 
    #         mesh_jnts, mesh_verts, mesh_faces = \
    #             run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
    #             betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)

    #         human_verts_list.append(mesh_verts)

    #     human_trans_list = torch.stack(human_trans_list)[0] # T X 3
    #     human_rot_list = torch.stack(human_rot_list)[0] # T X 22 X 3 X 3 
    #     human_jnts_list = torch.stack(human_jnts_list)[0, 0] # T X 22 X 3 
    #     human_verts_list = torch.stack(human_verts_list)[0, 0] # T X Nv X 3 
    #     human_faces_list = torch.stack(human_faces_list)[0].detach().cpu().numpy() # Nf X 3 

    #     obj_verts_list = torch.stack(obj_verts_list)[0] # T X Nv' X 3 
    #     obj_faces_list = np.asarray(obj_faces_list)[0] # Nf X 3

    #     actual_len_list = np.asarray(actual_len_list)[0] # scalar value 

    #     return human_trans_list, human_rot_list, human_jnts_list, human_verts_list, human_faces_list,\
    #     obj_verts_list, obj_faces_list, actual_len_list
    
    
    def plot_frame(self,ax, hand_position, gt_hand_position, pc, frame_idx, axis_limits):
        ax.clear()
        ax.set_xlim(axis_limits['x'])
        ax.set_ylim(axis_limits['y'])
        ax.set_zlim(axis_limits['z'])
        ax.set_title(f'Frame {frame_idx}')

        # Plot point cloud
        # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='b', marker='o')
        ax.scatter(pc[:, 2], pc[:, 0], pc[:, 1], s=1, c='b', marker='o')
        
        # Plot hand position
        for i in range(hand_position.shape[0]):
            ax.scatter(hand_position[i, 2], hand_position[i, 0], hand_position[i, 1], s=100, c='r', marker='o')
            
        for i in range(gt_hand_position.shape[0]):
            ax.scatter(gt_hand_position[i, 2], gt_hand_position[i, 0], gt_hand_position[i, 1], s=100, c='g', marker='o')
    
    # Plot hand position
    def create_gif(self,hand_positions, gt,pcs, gif_path, num_points=512):
        num_frames = hand_positions.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculate axis limits based on hand_position
        hand_positions_np = hand_positions.cpu().numpy()
        x_min, x_max = hand_positions_np[:, :, 2].min(), hand_positions_np[:, :, 2].max()
        y_min, y_max = hand_positions_np[:, :, 0].min(), hand_positions_np[:, :, 0].max()
        z_min, z_max = hand_positions_np[:, :, 1].min(), hand_positions_np[:, :, 1].max()
        padding = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)  # Add some padding for better visualization

        axis_limits = {
            'x': [x_min - padding, x_max + padding],
            'y': [y_min - padding, y_max + padding],
            'z': [z_min - padding, z_max + padding]
        }

        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame_idx in range(num_frames):
                hand_position = hand_positions[frame_idx].cpu().numpy()
                gt_hand_position = gt[frame_idx].cpu().numpy()
                # print(pcs.shape)
                pc = pcs[frame_idx].cpu().numpy()
                # print(pc.shape)
                pc_downsampled = pc
                
                self.plot_frame(ax, hand_position,gt_hand_position, pc_downsampled, frame_idx, axis_limits)
                
                # Save frame as image and append to GIF
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
                writer.append_data(image)
                
                
    def get_control_mask(self,shape, joint,ratio):
        mask = np.zeros(shape)
        mask = np.maximum(mask, self.get_global_joint_mask(shape, joint,ratio))
        return mask
    
    def select_random_indices(self ,bs, seq_len, num_selected):
        indices = []
        for _ in range(bs):
            indices.append(np.random.choice(seq_len, size=num_selected, replace=False))
        return np.array(indices)
        
    def compute_object_geo_bps(self, pc):
        
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        obj_trans = pc.mean(1) # T X 3
        
        # print(obj_trans.shape)
        # object_bps = self.compute_object_geo_bps(pc, obj_trans)
        bps_object_geo = self.bps_torch.encode(x=self.to_tensor(pc), \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo
        
    def to_tensor(self,array, dtype=torch.float32):
        if not torch.is_tensor(array):
            array = torch.tensor(array)
        return array.to(dtype)
    
    def get_global_joint_mask(self,shape, joint_index, ratio=1):
        """
        expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_joint (22 for HumanML3D), 3, seq_len))
        """
        bs, num_joint, joint_dim, seq_len = shape
        assert joint_dim == 3, "joint_dim must be 3, got {}".format(joint_dim)
        random_joint = (np.ones((1,bs), dtype=int) * joint_index)
        if np.abs(1 - ratio) < 1e-3:
            random_t = np.ones((bs, 1, 1, seq_len))
        else:
            num_selected = int(ratio * seq_len)
            random_t = np.zeros((bs, 1, 1, seq_len))
            selected_indices = self.select_random_indices(bs, seq_len, num_selected)
            random_t[np.arange(bs)[:, np.newaxis], :, :, selected_indices] = 1

        random_t = np.tile(random_t, (1, 1, 3, 1))
        mask = np.zeros(shape)
        for i in range(random_joint.shape[0]):
            mask[np.arange(bs)[:, np.newaxis], random_joint[i, :, np.newaxis], :, :] = random_t.astype(float)
        return mask
        
    def gen_vis_res_joint(self, all_res_list, data_dict, step, vis_gt=False, vis_tag=None):
        # all_res_list: BS X T X 12  
        lhand_color = np.asarray([255, 87, 51])  # red 
        rhand_color = np.asarray([17, 99, 226]) # blue
        lfoot_color = np.asarray([134, 17, 226]) # purple 
        rfoot_color = np.asarray([22, 173, 100]) # green 

        contact_pcs_colors = []
        contact_pcs_colors.append(lhand_color)
        contact_pcs_colors.append(rhand_color)
        contact_pcs_colors.append(lfoot_color)
        contact_pcs_colors.append(rfoot_color)
        contact_pcs_colors = np.asarray(contact_pcs_colors) # 4 X 3 
        
        seq_names = data_dict['seq_name'] # BS 
        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 

        # obj_rot = data_dict['obj_rot_mat'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 X 3
        obj_com_pos = data_dict['obj_com_pos'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 

        num_seq, num_steps, _ = all_res_list.shape
        
        # normalized_gt_hand_foot_pos = self.extract_palm_jpos_only_data(data_dict['joint']) 
        normalized_gt_hand_foot_pos = data_dict['gt_hands']#.reshape(-1, num_steps, 2, 3) 
        # Denormalize hand only 
        
        pred_hand_foot_pos = self.val_ds.de_normalize_jpos_min_max_hand_foot(all_res_list.cpu(), hand_only=True)
        gt_hand_foot_pos = self.val_ds.de_normalize_jpos_min_max_hand_foot(normalized_gt_hand_foot_pos.cpu(),hand_only=True) # BS X T X 2 X 3
        all_processed_hand_jpos = pred_hand_foot_pos.clone()#.reshape(-1, num_steps, 2, 3)  
        
        #print(num_seq)
        for seq_idx in range(num_seq): #bs
            if '_' not in seq_names:
                object_name = seq_names
            else:
                object_name = seq_names[seq_idx].split("_")[1]
            obj_mesh_verts = data_dict['pc'][seq_idx]
            curr_seq_pred_hand_foot_jpos = all_processed_hand_jpos[seq_idx]
            

        return all_processed_hand_jpos, gt_hand_foot_pos  

    def run_two_stage_pipeline(self): ##diffusion-sample for satge1+stage2
        
        fullbody_wdir = os.path.join(self.opt.project, self.opt.fullbody_exp_name, "weights")
        
        if not os.path.exists(f"./{opt.project}/{opt.exp_name}"):
            os.makedirs(f"./{opt.project}/{opt.exp_name}")

        logging.basicConfig(filename=f"./{opt.project}/{opt.exp_name}/eval.log", level=logging.INFO, 
                                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        #eval-stage1
        s1_lhand_jpe_per_seq = []
        s1_rhand_jpe_per_seq = []
        s1_hand_jpe_per_seq = [] 
        
        #eval-stage2
        hand_jpe_per_seq = []
        lhand_jpe_per_seq = []
        rhand_jpe_per_seq = []

        mpjpe_per_seq = []
        
        contact_precision_per_seq = []
        contact_recall_per_seq = [] 

        contact_acc_per_seq = []
        contact_f1_score_per_seq = [] 

        gt_contact_dist_per_seq = []
        contact_dist_per_seq = []

        sampled_all_res_per_seq = []

        foot_sliding = []
        gt_foot_sliding = []

        contact_percent_per_seq = []
        gt_contact_percent_per_seq = []
        
        self.joint_together = True

        #stage-1
        self.load(pretrained_path=self.opt.checkpoint)
        self.ema.ema_model.eval()
        

        with rich.progress.open('/storage/group/4dvlab/wangzy/SemGeoMo/data_pkl/omomo_fps15/tmp/test_hoi_motion.pkl','rb') as file:
            data_dict = pickle.load(file)
        text_dir = "/storage/group/4dvlab/congpsh/HHOI/OMOMO/pred_text/"
       
        s_idx = 0
        
        #random 100 samples
        name_list = ['sub17_woodchair_042', 'sub16_largebox_002', 'sub17_smalltable_002', 'sub17_suitcase_010', 'sub17_woodchair_048', 'sub16_clothesstand_015', 'sub16_trashcan_032', 'sub17_smalltable_036', 'sub17_suitcase_019', 'sub16_whitechair_031', 'sub16_plasticbox_038', 'sub16_largebox_004', 'sub17_tripod_024', 'sub17_suitcase_017', 'sub17_monitor_006', 'sub16_largebox_047', 'sub16_whitechair_002', 'sub16_largebox_014', 'sub17_smallbox_037', 'sub16_plasticbox_016', 'sub17_smallbox_038', 'sub16_whitechair_037', 'sub16_whitechair_015', 'sub17_smalltable_030', 'sub16_plasticbox_052', 'sub17_floorlamp_015', 'sub16_plasticbox_010', 'sub17_smalltable_013', 'sub17_woodchair_053', 'sub16_clothesstand_004', 'sub16_plasticbox_011', 'sub16_trashcan_020', 'sub17_woodchair_006', 'sub17_floorlamp_026', 'sub17_woodchair_019', 'sub17_smalltable_024', 'sub17_smalltable_009', 'sub17_smalltable_044', 'sub16_largebox_035', 'sub17_woodchair_055', 'sub17_monitor_003', 'sub17_tripod_004', 'sub16_plasticbox_034', 'sub17_smallbox_029', 'sub16_largebox_006', 'sub16_largebox_001', 'sub17_suitcase_024', 'sub17_floorlamp_013', 'sub16_largebox_042', 'sub17_woodchair_003', 'sub16_largebox_052', 'sub16_plasticbox_027', 'sub17_woodchair_037', 'sub16_largetable_030', 'sub16_largetable_013', 'sub17_smalltable_015', 'sub16_plasticbox_044', 'sub16_trashcan_015', 'sub17_monitor_028', 'sub17_smalltable_006', 'sub17_smallbox_031', 'sub17_smalltable_019', 'sub16_whitechair_028', 'sub16_plasticbox_020', 'sub17_tripod_025', 'sub16_trashcan_001', 'sub16_largetable_037', 'sub16_whitechair_020', 'sub17_smalltable_029', 'sub17_floorlamp_014', 'sub16_plasticbox_042', 'sub16_plasticbox_054', 'sub17_suitcase_007', 'sub17_monitor_026', 'sub17_smalltable_008', 'sub17_woodchair_014', 'sub16_largebox_013', 'sub17_monitor_010', 'sub17_floorlamp_032', 'sub16_largebox_051', 'sub17_smalltable_021', 'sub17_smallbox_021', 'sub16_plasticbox_036', 'sub17_smallbox_006', 'sub17_suitcase_023', 'sub17_tripod_028', 'sub16_whitechair_025', 'sub17_tripod_023', 'sub17_tripod_002', 'sub16_largebox_033', 'sub17_smalltable_035', 'sub17_smallbox_017', 'sub16_plasticbox_013', 'sub17_woodchair_011', 'sub17_woodchair_008', 'sub17_smallbox_004', 'sub16_plasticbox_043', 'sub17_woodchair_029', 'sub17_smalltable_014', 'sub16_largetable_016']
        print(len(name_list))

          
        while True:
            
            #print(s_idx)
            
            if(len(name_list)==0):
              break
              
            val_data_dict = next(self.val_dl)
            seq_name = val_data_dict["seq_name"] 
            name = seq_name[0]
            
            if name not in name_list:
                continue

            name_list.remove(name)
        
            obj_name = name.split("_")[1]

            print(name)
            print(len(name_list))
            s_idx += 1
            

            joint = data_dict[name]['joint'] #gt_frames
            frames = min(joint.shape[0],100) - 5
            
            #stage1---------------------------------------
            
            val_joint = val_data_dict['joint']
            bs, num_steps, _, _ = val_joint.shape 
            max_frames = num_steps
            gt_joint = val_joint.clone()
            gt_joint= self.val_ds.de_normalize_jpos_min_max(gt_joint.reshape(-1,22,3)).reshape(-1,22,3)

            lpalm_idx,rpalm_idx = 20,21
            joint_data = torch.Tensor(np.concatenate((val_joint[:,:, lpalm_idx].reshape(bs, num_steps,1,3), 
                                                val_joint[:,:, rpalm_idx].reshape(bs, num_steps,1,3)),axis=2))\
                                                    .reshape(bs, num_steps,-1).float().cuda()
                                                    
            dis_data = val_data_dict['dis'].reshape(bs, num_steps, -1).float().cuda()  
            if self.joint_together:
                val_data = torch.cat((joint_data,dis_data),-1)
                val_data_dict['gt_hands'] = joint_data

 
            obj_bps_data = val_data_dict['obj_bps'].reshape(bs, num_steps,-1).cuda() 
            obj_com_pos = val_data_dict['obj_com_pos'].cuda() # BS X T X 3 
            
            language = None
            if opt.text:
                #language = val_data_dict['text'] #gt_text
                
                # Read pred_text
                if not os.path.exists(f"{text_dir}/{name}.txt"):
                    continue
                with cs.open(f"{text_dir}/{name}.txt") as f:
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        t_tokens = line_split[1].split(' ')
                        
                        text_dict['caption'] = caption
                        text_dict['tokens'] = t_tokens
                language = text_dict['caption']
                
            val_data_dict['gt_dis'] = dis_data
            ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1).float() # BS X T X (3+1024*3)

            cond_mask = None 

            # Generate padding mask 
            actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
            tmp_mask = torch.arange(self.window+1).expand(val_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
            # BS X max_timesteps
            padding_mask = tmp_mask[:, None, :].to(val_data.device)

            max_num = 1
            
            # #print(val_data.shape)
            all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond, \
            cond_mask=cond_mask, padding_mask=padding_mask,text = language)  #BS X T X dim

            
            gt_dis = val_data_dict['gt_dis']
            pred_hand_foot_jpos = []
            
            vis_tag = "stage1_sample_"+str(s_idx)
            
            if self.joint_together:
                gt_dis = val_data_dict['gt_dis']
                pred_hand_foot_jpos, gt_hand_foot_pos = self.gen_vis_res_joint(all_res_list[:,:,:6], \
                    val_data_dict, 0, vis_tag=vis_tag)
            else: #only-joint
                pred_hand_foot_jpos, gt_hand_foot_pos = self.gen_vis_res_joint(all_res_list, \
                    val_data_dict, 0, vis_tag=vis_tag)
                    
            pred_hand = pred_hand_foot_jpos[0].clone()
            pred_hand_foot_jpos = pred_hand_foot_jpos[:,:frames,:,:]
            gt_hand_foot_pos = gt_hand_foot_pos[:,:frames,:,:]
            
            
            print(pred_hand_foot_jpos.shape)  
            # BS X T X 2 X 3   predicted hand pose 

            for s1_s_idx in range(bs): #bs
                s1_lhand_jpe, s1_rhand_jpe, s1_hand_jpe = compute_s1_metrics(pred_hand_foot_jpos[s1_s_idx, \
                    :actual_seq_len[s1_s_idx]], gt_hand_foot_pos[s1_s_idx, :actual_seq_len[s1_s_idx]])
                
                s1_lhand_jpe_per_seq.append(s1_lhand_jpe)
                s1_rhand_jpe_per_seq.append(s1_rhand_jpe)
                s1_hand_jpe_per_seq.append(s1_hand_jpe)
                
         
                
            print(np.asarray(s1_lhand_jpe_per_seq).mean())
            print(np.asarray(s1_rhand_jpe_per_seq).mean())
            print(np.asarray(s1_hand_jpe_per_seq).mean())
            
             
            #aff
            pred_aff = all_res_list[:,:,6:]
            pred_aff = pred_aff.reshape(bs, num_steps,1024,2)
            print(pred_aff.shape)
            
            
            #stage2---------------------------------------
            
            obj_rot = val_data_dict["obj_rot_mat"].float().reshape(-1,3,3)  #TX3X3
            obj_trans = val_data_dict["obj_trans"].float().reshape(-1,3)  #TX3X3
            obj_scale = val_data_dict["obj_scale"].float().reshape(-1,1)  #TX1
            
            sample = val_data_dict["motion"].float() ####bs(=1)xTX263
            samples=[]
            for i in range(1): 
              samples.append(sample) #1xbsxnfrmesx263
            sample=torch.stack([a for a in samples],dim=0).permute(0,3,1,2) #1 x 263 x bs x nframess
            input_motions = sample
            
            model_kwargs = dict()
            model_kwargs['y'] = dict()
            model_kwargs['y']['text'] = language #pred_text
            model_kwargs['y']['fine_text'] = val_data_dict['fine_text'] #fine_text
            
            
            # add inpainting mask according to args
            #control_joint = "all"
            #control_joint = ["left_foot","right_foot"]
            control_joint = ['left_wrist','right_wrist' ]  
            index_list = dict()
            index_list["target_contact"]=[]
            index_list["target_far"]=[]
            
            
            input_motions = self.data.dataset.t2m_dataset.inv_transform(input_motions.permute(0, 2, 3, 1).cpu()).float()
            global_joints =  recover_from_ric(input_motions, 22)
            global_joints = global_joints.view(-1, *global_joints.shape[2:]).permute(0, 2, 3, 1).to(dist_util.dev()) #1x22x3xframes
            global_joints = global_joints.to(dist_util.dev())
            global_joints.requires_grad = False
            ground_truth = global_joints.clone()
            global_joints[0,20:,:,:] = pred_hand.permute(1, 2, 0)  #use stage1-pred-hand
            #print(global_joints.shape) #1x22x3xnum_steps
            model_kwargs['y']['global_joint'] = global_joints 
            model_kwargs['y']['global_joint_mask'] = torch.tensor(get_control_mask("global_joint", global_joints.shape, joint = control_joint, ratio=1, dataset = self.dataset_name)).float().to(dist_util.dev())
            

            model_kwargs['y']['scale'] = torch.ones(self.batch_size, device=dist_util.dev()) * self.guidance_param
            
            model_kwargs['y']['pc'] = val_data_dict["pc"].to(dist_util.dev())
            model_kwargs['y']['bps'] = val_data_dict['obj_bps'].reshape(bs,num_steps,1024,3).to(model_kwargs['y']['pc'].device)
            
            model_kwargs['y']['dist'] = pred_aff.to(model_kwargs['y']['pc'].device) #model pred
            
            sample_fn = self.diffusion.p_sample_loop
            
            sample = sample_fn(
                self.model,
                (self.batch_size, self.model.njoints, self.model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                use_posterior = True,
                #aff = aff,
            )
            
            #print(sample.shape)  #1 x 263 x 1 x num_steps
            sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, 22)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) #1x22x3xnum_steps
            
            pointcloud = val_data_dict["pc"].reshape(-1,1024,3).cpu()[:frames]
            gt = ground_truth[0].cpu().numpy().transpose(2, 0, 1)[:frames]

            
            motion = sample[0].cpu().numpy().transpose(2, 0, 1)[:frames]
            print(motion.shape) #TX22X3
            print(pointcloud.shape)
            

            lhand_jpe, rhand_jpe, hand_jpe, mpjpe, gt_contact_dist, contact_dist, \
            contact_precision, contact_recall,contact_acc, contact_f1_score, \
            foot_sliding_jnts,gt_foot_sliding_jnts ,contact_percent,gt_contact_percent= compute_metrics(torch.tensor(gt),torch.tensor(motion),pointcloud)
                            
            hand_jpe_per_seq.append(hand_jpe)
            lhand_jpe_per_seq.append(lhand_jpe)
            rhand_jpe_per_seq.append(rhand_jpe) 

            mpjpe_per_seq.append(mpjpe)
            
            contact_precision_per_seq.append(contact_precision)
            contact_recall_per_seq.append(contact_recall) 

            contact_acc_per_seq.append(contact_acc) 
            contact_f1_score_per_seq.append(contact_f1_score) 

            gt_contact_dist_per_seq.append(gt_contact_dist)
            contact_dist_per_seq.append(contact_dist)

            foot_sliding.append(foot_sliding_jnts)
            gt_foot_sliding.append(gt_foot_sliding_jnts)

            contact_percent_per_seq.append(contact_percent)
            gt_contact_percent_per_seq.append(gt_contact_percent)
            
            print(np.asarray(lhand_jpe_per_seq).mean())
            print(np.asarray(rhand_jpe_per_seq).mean())
            print(np.asarray(hand_jpe_per_seq).mean())
            print(np.asarray(mpjpe_per_seq).mean())
            print( np.asarray(contact_precision_per_seq).mean())
            print(np.asarray(contact_recall_per_seq).mean())
            print(np.asarray(contact_acc_per_seq).mean())
            print(np.asarray(contact_f1_score_per_seq).mean())
            print(np.asarray(foot_sliding).mean())
            print(np.asarray(gt_foot_sliding).mean())
            print(np.asarray(contact_percent_per_seq).mean())
            print(np.asarray(gt_contact_percent_per_seq).mean())
            print("--------------------")
            
            #plot gif
            save_file = 'sample{:02d}_'.format(s_idx)+name+".gif"
            out_path = f"./{opt.project}/{opt.exp_name}/vis_gif"
            os.makedirs(out_path,exist_ok=True)
            animation_save_path = os.path.join(out_path, save_file)
            skeleton = paramUtil.t2m_kinematic_chain
            caption = model_kwargs['y']['text']
            inpainting_mask  = "global_joint"
            guidance = {'mask': model_kwargs['y']['global_joint_mask'][0], 'joint': gt}
            guidance['mask'] = guidance['mask'].cpu().numpy().transpose(2, 0, 1)[:frames]
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=self.data.dataset, fps=20, vis_mode=inpainting_mask,
                            gt_frames=[], joints2=gt, painting_features=inpainting_mask.split(','), guidance=guidance,pointcloud=pointcloud,person2=None,index_list =None)
                            
            #save npy
            out_npy_path = pjoin(out_path,"results/")
            os.makedirs(out_npy_path,exist_ok=True)
            npy_path = os.path.join(out_npy_path, name+'results.npy')
            print(f"saving results file to [{npy_path}]")
            np.save(npy_path,
                    {'motion': motion, 'text': caption, 'lengths': frames,
                     'num_samples': 1, 'num_repetitions': 1})



        if self.for_quant_eval:
            s1_mean_hand_jpe = np.asarray(s1_hand_jpe_per_seq).mean()
            s1_mean_lhand_jpe = np.asarray(s1_lhand_jpe_per_seq).mean()
            s1_mean_rhand_jpe = np.asarray(s1_rhand_jpe_per_seq).mean() 
  
            mean_hand_jpe = np.asarray(hand_jpe_per_seq).mean() 
            mean_lhand_jpe = np.asarray(lhand_jpe_per_seq).mean()
            mean_rhand_jpe = np.asarray(rhand_jpe_per_seq).mean()
            
            mean_mpjpe = np.asarray(mpjpe_per_seq).mean() 
            
            mean_contact_precision = np.asarray(contact_precision_per_seq).mean()
            mean_contact_recall = np.asarray(contact_recall_per_seq).mean() 
  
            mean_contact_acc = np.asarray(contact_acc_per_seq).mean()
            mean_contact_f1_score = np.asarray(contact_f1_score_per_seq).mean() 
  
            mean_gt_contact_dist = np.asarray(gt_contact_dist_per_seq).mean()
            mean_contact_dist = np.asarray(contact_dist_per_seq).mean()

            mean_gt_foot_fliding = np.asarray(gt_foot_sliding).mean()
            mean_foot_fliding = np.asarray(foot_sliding).mean()

            mean_gt_contact_percent = np.asarray(gt_contact_percent_per_seq).mean()
            mean_contact_percent = np.asarray(contact_percent_per_seq).mean()

  
            logging.info("*****************************************Quantitative Evaluation*****************************************")
            logging.info("The number of sequences: {0}".format(len(mpjpe_per_seq)))
            logging.info("Stage 1 Left Hand JPE: {0}, Stage 1 Right Hand JPE: {1}, Stage 1 Two Hands JPE: {2}".format(s1_mean_lhand_jpe, s1_mean_rhand_jpe, s1_mean_hand_jpe))
            logging.info("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            logging.info("MPJPE: {0}".format(mean_mpjpe))
            logging.info("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            logging.info("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score)) 
            logging.info("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))
            logging.info("Foot sliding: {0}, GT Foot sliding: {1}".format(mean_foot_fliding, mean_gt_foot_fliding))
            logging.info("Contact percent: {0}, GT Contact percent: {1}".format(mean_contact_percent, mean_gt_contact_percent))


def run_sample(opt, device, args,run_pipeline=False):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    args = args

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=100,
                              split='test',
                              load_mode='train',
                              size=args.num_samples)  # in train mode, you get both text and motion.
    
    # Define model 
    repr_dim = 2 * 3
    
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                objective="pred_x0", loss_type=loss_type, \
                batch_size=opt.batch_size,text = opt.text)

    diffusion_model.to(device)
        
                
    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=400000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False ,
        args = args,
        data = data
    )
    
    trainer.run_two_stage_pipeline() 

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='wandb_proj_name', help='wandb project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='stage1_exp_out', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--fullbody_exp_name', default='test_exp_with_eval', help='project/fullbody_exp_name')
    parser.add_argument('--fullbody_checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint') #stage1

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    # For running the whole pipeline. 
    parser.add_argument("--run_whole_pipeline", action="store_true")

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")

    parser.add_argument("--dataset_name", default=" ", help='root folder for dataset')

    parser.add_argument('--data_root_folder', default="/public/home/wangzy17/omomo_dataset/data", help='root folder for dataset')

    parser.add_argument('--datasettype',  default=None)
    parser.add_argument('--text',  default=False) #######
    
    parser.add_argument('--model_path', default='', help='model pt') ###stage2
    parser.add_argument("--use_posterior", action="store_true")
    parser.add_argument('--save_dir', default='', help='save to this dir')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    print(opt.text)
    args = edit_control_args()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    args.dataset = opt.dataset_name
    print(args.dataset)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device,args)
    elif opt.run_whole_pipeline:
        run_sample(opt, device, args,run_pipeline=True)