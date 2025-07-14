import argparse
import os
import numpy as np
import yaml
import random
import json 

import trimesh 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from pathlib import Path

import pickle as pkl


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import matplotlib.pyplot as plt
import random
import imageio

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from manip.data.hand_contact_data import HandContactDataset
from manip.model.transformer_hand_foot_manip_cond_diffusion_model import CondGaussianDiffusion 

from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion as FullBodyCondGaussianDiffusion

from eval_metric import compute_metrics, compute_s1_metrics, compute_collision

from matplotlib import pyplot as plt
# from utils.motion_process import recover_from_ric

import logging

def cycle(dl):   #来遍历dl中所有数据
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
        datasettype = None
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")
        self.predict_folder = results_folder.replace("weights", "")

        self.opt = opt 

        self.window = opt.window

        self.use_object_split = self.opt.use_object_split 

        self.data_root_folder = self.opt.data_root_folder 
        self.dataset_name = self.opt.dataset_name 

        self.prep_dataloader(window_size=opt.window)

        # self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_sample_res_on_train 

        self.add_hand_processing = self.opt.add_hand_processing 

        self.for_quant_eval = self.opt.for_quant_eval 

        self.use_gt_hand_for_eval = self.opt.use_gt_hand_for_eval 
        self.datasettype = self.opt.datasettype 
        self.joint_together = self.opt.joint_together

    def prep_dataloader(self, window_size):
        # Define dataset
        # train_dataset = HandContactDataset(train=True, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
        #     window=window_size, use_object_splits=self.use_object_split)
        val_dataset = HandContactDataset(train=False, data_root_folder=self.data_root_folder, dataset_name=self.dataset_name,\
            window=window_size, use_object_splits=self.use_object_split,debug=False)

        # self.ds = train_dataset 
        self.val_ds = val_dataset
        # self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
        #     shuffle=True, pin_memory=True, num_workers=0,drop_last=True))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=0,drop_last=False))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def prep_temporal_condition_mask(self, data, t_idx=0):
        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

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
            
            # print("=========",curr_seq_pred_hand_foot_jpos.shape,obj_mesh_verts.shape)
            
            os.makedirs(self.vis_folder,exist_ok=True)
            if seq_idx == 0: #vis
                vis_path = f"{self.vis_folder}/{seq_names[seq_idx]}_joint_output.gif"
                print("+++",vis_path)
                self.create_gif(curr_seq_pred_hand_foot_jpos,gt_hand_foot_pos[seq_idx], obj_mesh_verts, vis_path)

        return all_processed_hand_jpos, gt_hand_foot_pos  

    
    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        # print(f"Loaded weight: {weight_path}")

        # milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")

        print(self.opt.checkpoint)
        print(len(self.val_ds))
        print(self.vis_folder)
        
        self.load(weight_path,pretrained_path=self.opt.checkpoint)
        self.ema.ema_model.eval()

        num_sample = 4 # 100 --bs=32,32,32,4
        
        
        s1_lhand_jpe_per_seq = []
        s1_rhand_jpe_per_seq = []
        s1_hand_jpe_per_seq = [] 
        result = []
        
        mse_loss_list = []
        avg_cosine_sim_ = []
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    print("-----testing on training set")
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                val_joint = val_data_dict['joint']
                bs, num_steps, _, _ = val_joint.shape 
                lpalm_idx,rpalm_idx = 20,21
                joint_data = torch.Tensor(np.concatenate((val_joint[:,:, lpalm_idx].reshape(bs, num_steps,1,3), 
                                                    val_joint[:,:, rpalm_idx].reshape(bs, num_steps,1,3)),axis=2))\
                                                        .reshape(bs, num_steps,-1).float().cuda()
                # print(data_dict['dis'].shape,data_dict['dis'].dtype)        
                dis_data = val_data_dict['dis'].reshape(bs, num_steps, -1).float().cuda()  
                if self.joint_together:
                    val_data = torch.cat((joint_data,dis_data),-1)
                    val_data_dict['gt_hands'] = joint_data

     
                obj_bps_data = val_data_dict['obj_bps'].reshape(bs, num_steps,-1).cuda() 
                obj_com_pos = val_data_dict['obj_com_pos'].cuda() # BS X T X 3 
                language = None
                if opt.text:
                    language = val_data_dict['text']
                # print(val_data_dict.keys())
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

                #print(val_data.shape)

                all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond, \
                cond_mask=cond_mask, padding_mask=padding_mask,text = language)  #BS X T X dim

                vis_tag = "stage1_sample_"+str(s_idx)
# 
                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"
                
                gt_dis = val_data_dict['gt_dis']
                pred_hand_foot_jpos = []
                if self.joint_together:
                    # all_res_list = all_res_list[:,:,6:]
                    gt_dis = val_data_dict['gt_dis']
                    pred_hand_foot_jpos, gt_hand_foot_pos = self.gen_vis_res_joint(all_res_list[:,:,:6], \
                        val_data_dict, 0, vis_tag=vis_tag)
                else: #only-joint
                    pred_hand_foot_jpos, gt_hand_foot_pos = self.gen_vis_res_joint(all_res_list, \
                        val_data_dict, 0, vis_tag=vis_tag)

                bs_s1_hand_jpe = []
                for s1_s_idx in range(bs):
                    s1_lhand_jpe, s1_rhand_jpe, s1_hand_jpe = compute_s1_metrics(pred_hand_foot_jpos[s1_s_idx, \
                        :actual_seq_len[s1_s_idx]], gt_hand_foot_pos[s1_s_idx, :actual_seq_len[s1_s_idx]])
                    
                    s1_lhand_jpe_per_seq.append(s1_lhand_jpe)
                    s1_rhand_jpe_per_seq.append(s1_rhand_jpe)
                    s1_hand_jpe_per_seq.append(s1_hand_jpe)
                    bs_s1_hand_jpe.append(s1_hand_jpe) 

                if self.joint_together:
                    all_res_list = all_res_list[:,:,6:]
                # print(all_res_list.shape,val_data_dict.shape)
                mse_loss = F.mse_loss(all_res_list, gt_dis)
                mse_loss_list.append(mse_loss.item())
                print(f"MSE Loss: {mse_loss.item():.4f}")

                cosine_sim = cosine_similarity(all_res_list, gt_dis, dim=-1)  # 在点维度计算
                avg_cosine_sim = cosine_sim.mean()  # 平均余弦相似度
                print(f"Average Cosine Similarity: {avg_cosine_sim.item():.4f}")
                avg_cosine_sim_.append(avg_cosine_sim)
                
                M = len(s1_lhand_jpe_per_seq)
                # s1_lhand_jpe_per_seq = np.array(s1_lhand_jpe_per_seq)
                # s1_rhand_jpe_per_seq = np.array(s1_rhand_jpe_per_seq)
                # s1_hand_jpe_per_seq = np.array(s1_lhand_jpe_per_seq)
                print(np.array(s1_lhand_jpe_per_seq).mean(),np.array(s1_rhand_jpe_per_seq).mean(),np.array(s1_hand_jpe_per_seq).mean())
                logging.info("Stage 1 Left Hand JPE: {0}, Stage 1 Right Hand JPE: {1}, Stage 1 Two Hands JPE: {2}".format(sum(s1_lhand_jpe_per_seq)/M, sum(s1_rhand_jpe_per_seq)/M, sum(s1_hand_jpe_per_seq)/M))
                
                result_each = {
                    'pred_aff':all_res_list,
                    'data_info':val_data_dict,
                    'pred':pred_hand_foot_jpos,
                }
                result.append(result_each)
                    # print(s1_lhand_jpe)
        
        # with open(f'{self.predict_folder}result_aff2.pkl', 'wb') as handle:
        #     pkl.dump(result, handle)
        mse_loss = sum(mse_loss_list[:-1])/(len(mse_loss_list)-1)
        print(f"Avg MSE Loss: {mse_loss :.4f}")
        avg_cosine_sim = sum(avg_cosine_sim_[:-1])/(len(avg_cosine_sim_)-1)
        print(f"Cosine Similarity: {avg_cosine_sim :.4f}")
             

    def extract_palm_jpos_only_data(self, data_input):
        # data_input: BS X T X D (22*3+22*6)
        lpalm_idx = 22   ##22
        rpalm_idx = 23 #####23
        data_input = torch.cat((data_input[:, :, lpalm_idx*3:lpalm_idx*3+3], \
                data_input[:, :, rpalm_idx*3:rpalm_idx*3+3]), dim=-1)
        # BS X T X (2*3)

        return data_input 

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

    def process_hand_foot_contact_jpos(self, hand_foot_jpos, object_mesh_verts, obj_rot):
        # hand_foot_jpos: T X 2 X 3 
        # object_mesh_verts: T X Nv X 3 
        # object_mesh_faces: Nf X 3 
        # obj_rot: T X 3 X 3 
        all_contact_labels = []
        all_object_c_idx_list = []
        all_dist = []

        obj_rot = torch.from_numpy(obj_rot).to(hand_foot_jpos.device)
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

    def downsample_pc(self,pc, num_points):
        indices = random.sample(range(pc.shape[0]), num_points)
        return pc[indices]

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
        x_min, x_max = hand_positions_np[:, :, 0].min(), hand_positions_np[:, :, 0].max()
        y_min, y_max = hand_positions_np[:, :, 1].min(), hand_positions_np[:, :, 1].max()
        z_min, z_max = hand_positions_np[:, :, 2].min(), hand_positions_np[:, :, 2].max()
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
                if pc.shape[0] > num_points:  
                    pc_downsampled = self.downsample_pc(pc, num_points)
                else:
                    pc_downsampled = pc
                
                self.plot_frame(ax, hand_position,gt_hand_position, pc_downsampled, frame_idx, axis_limits)
                
                # Save frame as image and append to GIF
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
                writer.append_data(image)
                
    def create_dis_gif(self, dis, pcs, gif_path):
        # print(dis.shape, pcs.shape)
        t, c = dis.shape
        pcs_yxz = torch.zeros_like(pcs)
        pcs_yxz[:,:,2] = pcs[:,:,1]
        pcs_yxz[:,:,1] = pcs[:,:,0]
        pcs_yxz[:,:,0] = pcs[:,:,2]
        data_distances = dis.reshape(t, -1, 2)
        data_positions = pcs_yxz  # 绝对位置数据, T*N*3

        # Calculate the min and max values for the positions to center the point cloud
        min_vals = torch.min(data_positions.reshape(-1, 3), dim=0).values
        max_vals = torch.max(data_positions.reshape(-1, 3), dim=0).values
        center = (max_vals + min_vals) / 2
        max_range = torch.max(max_vals - min_vals) / 2

        # Set the limits for the axes based on the calculated center and range
        xlim = [center[0] - max_range, center[0] + max_range]
        ylim = [center[1] - max_range, center[1] + max_range]
        zlim = [center[2] - max_range, center[2] + max_range]

        # 创建图形和3D轴
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        def update(frame):
            # 清除当前内容
            ax1.clear()
            ax2.clear()
            # 获取当前帧的位置和距离数据
            positions = data_positions[frame]
            distances1 = data_distances[frame, :, 0]
            distances2 = data_distances[frame, :, 1]

            # 绘制第一个子图
            sc1 = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=distances1, cmap='hot', s=25)
            ax1.set_title('Distance Set 1')
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax1.set_zlim(zlim)

            # 绘制第二个子图
            sc2 = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=distances2, cmap='hot', s=25)
            ax2.set_title('Distance Set 2')
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            ax2.set_zlim(zlim)

            return sc1, sc2

        # 创建动画
        ani = FuncAnimation(fig, update, frames=t, blit=False)

        # 保存动画为GIF
        ani.save(gif_path, writer='imagemagick', fps=10)

        # 关闭图表
        plt.close()
                
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
        
        seq_names = data_dict['seq_name'] # BS 
        seq_len = data_dict['seq_len'].detach().cpu().numpy() # BS 

        # obj_rot = data_dict['obj_rot_mat'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 X 3
        obj_com_pos = data_dict['obj_com_pos'][:all_res_list.shape[0]].to(all_res_list.device) # BS X T X 3 

        num_seq, num_steps, _ = all_res_list.shape
        
        normalized_gt_hand_foot_pos = self.extract_palm_jpos_only_data(data_dict['joint']) 
        gt_dis = data_dict['gt_dis']#.reshape(-1, num_steps, 2, 3) 
        if gt_dis.shape[-1] > 2048:
            gt_dis = gt_dis[:,:,6:]
        pred_dis = all_res_list.cpu()
        # Denormalize hand only 

        for seq_idx in range(num_seq):
            object_name = seq_names[seq_idx].split("_")[1]
            obj_mesh_verts = data_dict['pc'][seq_idx]
            if self.dataset_name == 'behave':
                obj_mesh_verts = obj_mesh_verts[...,::2,:]

            os.makedirs(self.vis_folder,exist_ok=True)
            if seq_idx < 50:
                vis_path = f"{self.vis_folder}/{seq_names[seq_idx]}_output.gif"
            # ------------GIF
                # print(curr_seq_pred_hand_foot_jpos.shape,gt_hand_foot_pos[seq_idx].shape)
                self.create_dis_gif(pred_dis[seq_idx], obj_mesh_verts, vis_path)
                vis_path = f"{self.vis_folder}/{seq_names[seq_idx]}_gt.gif"
                self.create_dis_gif(gt_dis[seq_idx].cpu(), obj_mesh_verts, vis_path)
                print("The vis path is ", vis_path)
              
            # if vis_tag is None:
            #     dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            # else:
            #     dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
                
            
        # if self.use_gt_hand_for_eval:
        #     all_processed_hand_jpos = self.ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.cuda())
        # else:
        #     all_processed_hand_jpos = self.ds.normalize_jpos_min_max_hand_foot(all_processed_hand_jpos) # BS X T X 4 X 3 

        # gt_hand_foot_pos = self.ds.normalize_jpos_min_max_hand_foot(gt_hand_foot_pos.cuda())
       

def run_sample(opt, device, run_pipeline=False):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 2 * 1024 #aff
    if opt.joint_together:
        repr_dim = 2 * 3  # for seperate joint+aff diffusion
    
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
        use_wandb=False 
    )
    
    
    trainer.cond_sample_res()

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

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")
    parser.add_argument('--joint_together',  default=False)

    # For testing sampled results on training dataset 
    parser.add_argument("--test_sample_res_on_train", action="store_true")

    # For running the whole pipeline. 
    parser.add_argument("--run_whole_pipeline", action="store_true")
    

    parser.add_argument("--add_hand_processing", action="store_true")

    parser.add_argument("--for_quant_eval", action="store_true")

    parser.add_argument("--use_gt_hand_for_eval", action="store_true")

    parser.add_argument("--use_object_split", action="store_true")
    parser.add_argument("--dataset_name", default="omomo", help='root folder for dataset')

    parser.add_argument('--data_root_folder', default="/storage/group/4dvlab/OMOMO/", help='root folder for dataset')

    parser.add_argument('--datasettype',  default=None)
    parser.add_argument('--text',  default=False)
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    
    opt = parse_opt()
    # print(opt.data_root_folder)
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    logging.basicConfig(filename=opt.save_dir + '/sample.log', level=logging.INFO)
    print(opt.save_dir)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    elif opt.run_whole_pipeline:
        run_sample(opt, device, run_pipeline=True)
    