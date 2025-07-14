import itertools
import argparse
import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from diffusion.control_diffusion import ControlGaussianDiffusion
#from diffusion.two_person_control_diffusion import TwoPeopleControlGaussianDiffusion
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.metrics import calculate_skating_ratio, compute_kps_error, compute_kps_error_with_distance, calculate_trajectory_error
from data_loaders.humanml.data.dataset import abs3d_to_rel, sample_to_motion

import joblib 
import os
from matplotlib import pyplot as plt
import imageio


from ...humanml_utils import get_control_mask, HML_JOINT_NAMES, load_json_file
from ema_pytorch import EMA
import sys
sys.path.append(".....")
from manip.model.transformer_hand_foot_manip_cond_diffusion_model import CondGaussianDiffusion 
from eval_metric import compute_metrics, compute_s1_metrics
sys.path.append("....")
from utils.sampling_utils import double_take_arb_len, unfold_sample_arb_len
from utils import dist_util
from data_loaders.humanml.common.skeleton import Skeleton
#from data_loaders.humanml.utils.paramUtil import paramUtil
#from data_loaders.humanml.utils.plot_script import plot_3d_motion


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint') #stage1
    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    parser.add_argument('--text',  default=True) #######
    parser.add_argument('--model_path',default="") #stage2
    parser.add_argument('--dataset',default= "omomo" )
    parser.add_argument('--replication_times',default=10)
    parser.add_argument('--mask_ratio',default= 1 )
    parser.add_argument('--bfgs_times_first',default= 5)
    parser.add_argument('--control_joint ',default=["left_wrist","right_wrist"])
    parser.add_argument('--bfgs_times_last',default= 10)
    parser.add_argument('--bfgs_interval',default= 1)
    parser.add_argument("--use_posterior", action="store_true")
    opt = parser.parse_args()
    return opt
    
    
    
def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator

class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'is_transition': model_kwargs['y']['is_transition'][bs_i]
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, transitions = data['motion'], data['length'], data['caption'], data['tokens'], data['is_transition']
        sent_len = data['cap_len']

        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # FIXME: I deleted the extra return value ([]), should check whether it breakes anything or not
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []


class CompMDMControlGeneratedDataset(CompMDMGeneratedDataset):  ##we use-

    def relative_motion_to_global_joint(self, motion):
        n_joints = 22 if motion.shape[1] == 263 else 21
        if n_joints == 22:
            dataset_name = 'omomo'  #####
            print(self.dataset)
        elif n_joints == 21:
            dataset_name = 'kit'
        else:
            raise ValueError(f"n_joints must be 21 or 22, got {n_joints}")
        unnormalized_motion = self.dataset.t2m_dataset.inv_transform_torch(motion.permute(0, 2, 3, 1)).float()
        global_joints = recover_from_ric(unnormalized_motion, n_joints)
        global_joints = global_joints.view(-1, *global_joints.shape[2:]).permute(0, 2, 3, 1)
        return global_joints, dataset_name
        
    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X 22/24 X 3 
        
        # print((self.global_jpos_max.shape,self.global_jpos_min.shape))
        normalized_jpos = (ori_jpos - self.global_jpos_min)/(self.global_jpos_max-self.global_jpos_min)
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        return normalized_jpos # T X 22/24 X 3 
    
    def de_normalize_jpos_min_max(self, normalized_jpos):
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range
        de_jpos = normalized_jpos * (self.global_jpos_max-self.global_jpos_min) + self.global_jpos_min

        return de_jpos # T X 22/24 X 3

    def normalize_jpos_min_max_hand_foot(self, ori_jpos, hand_only=True):
        # ori_jpos: BS X T X 2 X 3 
        lhand_idx = 20
        rhand_idx = 21

        lfoot_idx = 10
        rfoot_idx = 11 

        bs = ori_jpos.shape[0] 
        num_steps = ori_jpos.shape[1] 
        ori_jpos = ori_jpos.reshape(bs, num_steps, -1) # BS X T X (2*3)

        if hand_only:
            hand_foot_jpos_max = np.concatenate((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx]), 0) # (3*4)

            hand_foot_jpos_min = np.concatenate((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx]), 0)
        # else:
        #     hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
        #             self.global_jpos_max[0, rhand_idx], \
        #             self.global_jpos_max[0, lfoot_idx], \
        #             self.global_jpos_max[0, rfoot_idx]), dim=0) # (3*4)

        #     hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
        #             self.global_jpos_min[0, rhand_idx], \
        #             self.global_jpos_min[0, lfoot_idx], \
        #             self.global_jpos_min[0, rfoot_idx]), dim=0)

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]
        normalized_jpos = (ori_jpos - hand_foot_jpos_min)/(hand_foot_jpos_max - hand_foot_jpos_min)
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 

        normalized_jpos = normalized_jpos.reshape(bs, num_steps, -1, 3)

        return normalized_jpos # BS X T X 2 X 3 
    
    def de_normalize_jpos_min_max_hand_foot(self, normalized_jpos, hand_only=True):
        # normalized_jpos: BS X T X (3*4)
        lhand_idx = 20
        rhand_idx = 21
       
        lfoot_idx = 10
        rfoot_idx = 11 

        bs, num_steps, _ = normalized_jpos.shape 

        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

        if hand_only:
        
            hand_foot_jpos_max = np.concatenate((self.global_jpos_max[0, lhand_idx], \
                    self.global_jpos_max[0, rhand_idx]), 0) # (3*4)

            hand_foot_jpos_min = np.concatenate((self.global_jpos_min[0, lhand_idx], \
                    self.global_jpos_min[0, rhand_idx]), 0)
        # else:
        #     hand_foot_jpos_max = torch.cat((self.global_jpos_max[0, lhand_idx], \
        #             self.global_jpos_max[0, rhand_idx], \
        #             self.global_jpos_max[0, lfoot_idx], \
        #             self.global_jpos_max[0, rfoot_idx]), dim=0) # (3*4)

        #     hand_foot_jpos_min = torch.cat((self.global_jpos_min[0, lhand_idx], \
        #             self.global_jpos_min[0, rhand_idx], \
        #             self.global_jpos_min[0, lfoot_idx], \
        #             self.global_jpos_min[0, rfoot_idx]), dim=0)

        hand_foot_jpos_max = hand_foot_jpos_max[None, None]
        hand_foot_jpos_min = hand_foot_jpos_min[None, None]

        de_jpos = normalized_jpos * (hand_foot_jpos_max-hand_foot_jpos_min) + hand_foot_jpos_min

        return de_jpos.reshape(bs, num_steps, -1, 3) # BS X T X 4(2) X 3 
        
        
    def gen_vis_res(self, all_res_list):
        # all_res_list: BS X T X 12  
        num_seq, num_steps, _ = all_res_list.shape
        
        # Denormalize hand only 
        pred_hand_foot_pos = self.de_normalize_jpos_min_max_hand_foot(all_res_list.cpu(), hand_only=True)
        all_processed_hand_jpos = pred_hand_foot_pos.clone() 
        return all_processed_hand_jpos  
        
    def load(self, pretrained_path=None):
        if pretrained_path != '':
            data = torch.load(pretrained_path)
            self.ema.load_state_dict(data['ema'], strict=False)
        else:
            pass
            
    
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
    def create_gif(self,hand_positions, gt,pcs, gif_path, num_points=1024):
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
        
        
    def __init__(self, args, model, diffusion : ControlGaussianDiffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)

        print(self.dataset)
        
        print(args.dataset)
        # print(args.use_posterior)
        
        if args.dataset == "omomo":
          min_max_mean_std_data_path = "/storage/group/4dvlab/wangzy/SemGeoMo/data_pkl/omomo_fps15/min_max_mean_std_data_window_100_cano_joints24.p"
          
        print(min_max_mean_std_data_path)
        
        min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        self.global_jpos_min = min_max_mean_std_jpos_data['global_jpos_min'].reshape(1,22, 3)
        self.global_jpos_max = min_max_mean_std_jpos_data['global_jpos_max'].reshape(1,22, 3)
        
        opt = parse_opt()
        opt.text = True
        self.opt = opt
        # Define model  stge1
        repr_dim = 1024 * 2 + 2 * 3  
        loss_type = "l1"
        diffusion_model = CondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model, \
                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                    max_timesteps=max_motion_length+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    batch_size=opt.batch_size,text = opt.text)
        diffusion_model.to("cuda:0")
        
        
        ema_decay=0.995
        ema_update_every=10
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        
        
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()
        control_joint = args.control_joint
        trained_joint = args.model_path.split('/')[-2].split('_')[-1]
        #if control_joint == 'pelvis': 
            #assert trained_joint == 'pelvis', f"control_joint is pelvis but trained_joint is {trained_joint}"
        #else:
            #assert trained_joint == 'all'
            #assert control_joint in HML_JOINT_NAMES or control_joint == 'all' or control_joint == 'random_two' \
            #or control_joint == 'random_three', f"control_joint must be one of {HML_JOINT_NAMES} or 'all' or 'random_two' or 'random_three', got ###
        control_joint = ["left_wrist","right_wrist"]
        print(f"control_joint: {control_joint}")

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale
                
                global_joints, dataset_name = self.relative_motion_to_global_joint(motion)
                global_joints.requires_grad = False
                
                #---------------------------------------
                
                self.load(pretrained_path=self.opt.checkpoint)
                self.ema.ema_model.eval()
    
                
                pcs = model_kwargs['y']['pc'] 
                pcs = torch.stack([torch.tensor(arr).float() for arr in pcs])#bsxTx1024x3
                bs, num_steps, _, _ = pcs.shape 
                obj_com_pos = torch.zeros((bs,num_steps,3)).cuda() 
                print(global_joints.shape)
                gt_joint = np.array(global_joints.clone().permute(0,3,1,2)) #gt_joint  bsxTx22x3
                
                #plot gif(check gt)
                # save_file = "1.gif"
                # out_path = "/storage/group/4dvlab/wangzy/uni_regen/exp-1/omomo-stage2/vis/"
                # os.makedirs(out_path,exist_ok=True)
                # animation_save_path = os.path.join(out_path, save_file)
                # skeleton = paramUtil.t2m_kinematic_chain
                # caption = "aaaa"
                # inpainting_mask  = "global_joint"
                # guidance = None
                # plot_3d_motion(animation_save_path, skeleton, gt_joint[1], title=caption,
                #                 dataset=None, fps=20, vis_mode=inpainting_mask,
                #                 gt_frames=[], joints2=gt_joint[1], painting_features=inpainting_mask.split(','), guidance=guidance,pointcloud=pcs[1],person2=None,index_list =None)
                                
                                
                actual_seq_len = torch.zeros((bs))
                for i in range(0,bs):
                  joint = gt_joint[i]  #TX22X3
                  pc = pcs[i]
                  val_joint = joint.reshape(-1,22,3)
                  val_joint = self.normalize_jpos_min_max(val_joint[:, :22*3].reshape(-1, 22, 3)) # T X 22 X 3 norm jpos
                  gt_joint[i]  = val_joint
                  pcs[i] = pc
                  obj_com_pos[i] = pc.mean(1)
                  actual_seq_len[i]=torch.tensor(num_steps)
                
                max_frames = num_steps
                lpalm_idx,rpalm_idx = 20,21
                joint = torch.Tensor(np.concatenate((gt_joint[:,:, lpalm_idx].reshape(bs, num_steps,1,3), 
                                                    gt_joint[:,:, rpalm_idx].reshape(bs, num_steps,1,3)),axis=2))\
                                                        .reshape(bs, num_steps,-1).float().cuda()                      
                dist = model_kwargs['y']['dist']
                dist  =  torch.stack([torch.tensor(arr).float() for arr in dist])
                dist = dist.reshape(bs, num_steps, -1).float().cuda()
                bps = model_kwargs['y']['bps']
                bps  =  torch.stack([torch.tensor(arr).float() for arr in bps])
                obj_bps_data = bps.reshape(bs, num_steps,-1).cuda() 
                language = model_kwargs['y']['text']
                    
                ori_data_cond = torch.cat((obj_com_pos, obj_bps_data), dim=-1).float() # BS X T X (3+1024*3)
                val_data = torch.cat((joint,dist),-1)
                
                cond_mask = None 
    
                # Generate padding mask 
                actual_seq_len = actual_seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
                tmp_mask = torch.arange(num_steps+1).expand(val_data.shape[0], \
                num_steps+1) < actual_seq_len[:, None].repeat(1, num_steps+1)  
                # BS X max_timesteps
                padding_mask = tmp_mask[:, None, :].to(val_data.device)
    
                #hand
                all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond, \
                cond_mask=cond_mask, padding_mask=padding_mask,model_kwargs=None,text = language)
                
    
                pred_hand_foot_jpos = self.gen_vis_res(all_res_list[:,:,:6]) #joint
                pred_hand = pred_hand_foot_jpos.clone()
                gt_hand = self.de_normalize_jpos_min_max_hand_foot(joint.cpu(), hand_only=True)
                
                
                # vis_folder = "/storage/group/4dvlab/wangzy/uni_regen/exp-1/omomo-stage2/vis/"
                # os.makedirs(vis_folder,exist_ok=True)
                # for seq_idx in range(1,2):
                #   vis_path = f"{vis_folder}/{seq_idx}_output.gif"
                #   self.create_gif(pred_hand_foot_jpos[seq_idx],gt_hand[seq_idx], pcs[seq_idx], vis_path)
                        
                #s1_lhand_jpe_per_seq = []
                #s1_rhand_jpe_per_seq = []
                #s1_hand_jpe_per_seq = [] 
                #for s1_s_idx in range(bs): #bs
                #  s1_lhand_jpe, s1_rhand_jpe, s1_hand_jpe = compute_s1_metrics(pred_hand_foot_jpos[s1_s_idx], gt_hand[s1_s_idx])
                #  s1_lhand_jpe_per_seq.append(s1_lhand_jpe)
                #  s1_rhand_jpe_per_seq.append(s1_rhand_jpe)
                #  s1_hand_jpe_per_seq.append(s1_hand_jpe)
                #print(np.asarray(s1_lhand_jpe_per_seq).mean())
                #print(np.asarray(s1_rhand_jpe_per_seq).mean())
                #print(np.asarray(s1_hand_jpe_per_seq).mean())
               
                #aff
                pred_aff = all_res_list[:,:,6:]
                pred_aff = pred_aff.reshape(bs, num_steps,1024,2)
                
                #---------------------------------------------
                
                #global_joints[:,20:,:,:] = pred_hand.permute(0, 2, 3, 1)  #use pred-hand
                model_kwargs['y']['dist'] = pred_aff.to(pcs.device) #model pred-aff
                model_kwargs['y']['global_joint'] = global_joints.to(dist_util.dev())
                model_kwargs['y']['global_joint_mask'] = torch.tensor(get_control_mask(args.inpainting_mask, shape = global_joints.shape, joint = control_joint, ratio = args.mask_ratio, dataset = dataset_name)).bool().to(dist_util.dev())
               
                
                

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    sample = sample_fn(                #####sample
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=motion.to(dist_util.dev()),
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        use_posterior = args.use_posterior,
                        pointcloud = torch.tensor(model_kwargs['y']['pc'])
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    sample = sample[:, :, :, :self.max_motion_length]
                    cur_global_joints, _ = self.relative_motion_to_global_joint(sample)
                    skate_ratio, _ = calculate_skating_ratio(cur_global_joints)  # [batch_size]
                    kps_error, t_mask = compute_kps_error(cur_global_joints, model_kwargs['y']['global_joint'], model_kwargs['y']['global_joint_mask'])  # output is [bs, seq_len]
                    traj_errors = calculate_trajectory_error(kps_error, t_mask)
                    
                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'dist_error': kps_error[bs_i],
                                    'skate_ratio': skate_ratio[bs_i],
                                    'traj_errors': traj_errors[bs_i],
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]
                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        skate_ratio = data['skate_ratio']
        dist_error = data['dist_error']
        traj_errors = data['traj_errors']
        
        print(self.dataset.dataset_name)
            

        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), skate_ratio, traj_errors

class CompMDMInpaintingGeneratedDataset(CompMDMGeneratedDataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale
                
                model_kwargs['y']['inpainted_motion'] = motion.to(dist_util.dev())
                model_kwargs['y']['inpainting_mask'] = torch.tensor(get_control_mask(args.inpainting_mask, motion.shape)).float().to(dist_util.dev())

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = sample_fn(
                        model,
                        motion.shape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=motion.to(dist_util.dev()),
                        progress=True,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'is_transition': model_kwargs['y']['is_transition'][bs_i]
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer



class CompMDMUnfoldingGeneratedDataset2(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., num_unfoldings=10):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length

        self.num_steps_to_generate = num_unfoldings

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()

        n_frames = 100  # FIXME - using fixed length

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                # tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = []
                    all_texts = []
                    all_lengths = []
                    all_tokens = []

                    for bs_i in range(dataloader.batch_size):
                        _tokens = [model_kwargs['y']['tokens'][bs_i - ii].split('_') for ii in
                                   reversed(range(self.num_steps_to_generate))]
                        texts = [model_kwargs['y']['text'][bs_i - ii] for ii in
                                 reversed(range(self.num_steps_to_generate))]
                        lengths = [n_frames - args.handshake_size] * self.num_steps_to_generate
                        # lengths = [model_kwargs['y']['lengths'][bs_i - ii] for ii in
                        #            reversed(range(self.num_steps_to_generate))]
                        all_texts.append(texts)
                        all_lengths.append(lengths)
                        all_tokens.append(_tokens)

                    new_model_kwargs = {
                        'y': {
                            'text': list(itertools.chain.from_iterable(all_texts)),
                            'lengths': list(itertools.chain.from_iterable(all_lengths)),
                            # TODO - support more than one sample in batch
                        }
                    }
                    new_batch_size = len(new_model_kwargs['y']['text'])

                    transition = torch.zeros(n_frames)
                    transition[:args.handshake_size] = 1.
                    transition[-args.handshake_size:] = 1.
                    transition = torch.tile(transition.unsqueeze(0), dims=(new_batch_size, 1))
                    transition[0, :args.handshake_size] = 0
                    transition[-1, -args.handshake_size:] = 0
                    new_model_kwargs['y']['is_transition'] = transition

                    # add CFG scale to batch
                    if scale != 1.:
                        new_model_kwargs['y']['scale'] = torch.ones(new_batch_size, device=dist_util.dev()) * scale
                    samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, new_model_kwargs,
                                                                     n_frames=n_frames, eval_mode=True) # TODO: check if possible using Doubletake arblen instead
                    all_samples = samples_per_rep_list[0]  # we only do one rep
                    sample = [[all_samples[bs_i*self.num_steps_to_generate + step_i, :, :, args.handshake_size:].squeeze().permute(1,0).cpu().numpy() for step_i in range(self.num_steps_to_generate)] for bs_i in range(dataloader.batch_size)]

                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i],
                                    'length': all_lengths[bs_i],
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i],
                                        'length': all_lengths[bs_i],
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'][self.step_to_eval], data['length'][self.step_to_eval], data['caption'][self.step_to_eval], data['tokens'][self.step_to_eval]
        sent_len = data['cap_len'][self.step_to_eval]

        if self.transition:
            max_tran_length = 40
            prev_motion = data['motion'][self.step_to_eval-1]
            cur_motion = data['motion'][self.step_to_eval]

            prev_motion_end = data['length'][self.step_to_eval-1]
            prev_motion_start = max(prev_motion_end - (max_tran_length//2), 0)

            cur_motion_start = 0
            cur_motion_end = min(max_tran_length // 2, data['length'][self.step_to_eval])

            # print(f'prev motion [{prev_motion_start}:{prev_motion_end}]')
            # print(f'cur motion [{cur_motion_start}:{cur_motion_end}]')

            motion = np.concatenate([prev_motion[prev_motion_start:prev_motion_end],
                                     cur_motion[cur_motion_start:cur_motion_end]], axis=0)
            m_length = motion.shape[0]
            # print(f'transition length [{motion.shape[0]}], max is [{max_tran_length}]')
            pad = np.zeros((self.max_motion_length - motion.shape[0], prev_motion.shape[1]), dtype=prev_motion.dtype)
            motion = np.concatenate([motion, pad], axis=0)
            assert motion.shape[0] == self.max_motion_length, f'motion.shape[0]={motion.shape[0]}'


        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []

import  numpy as np
def pad_sample_with_zeros(sample, max_len=250):
    # pad inp, change lenghts, and pad is transition
    seq_len, n_feats = sample.shape
    len_to_pad = max_len - seq_len
    np.zeros_like(sample)
    sample_padding = np.zeros((len_to_pad, n_feats))
    sample = np.concatenate((sample, sample_padding))
    return sample

class CompMDMUnfoldingGeneratedDataset(Dataset):
    def __init__(self, args, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., num_unfoldings=10):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False
        clip_denoised = False
        self.max_motion_length = max_motion_length

        self.num_steps_to_generate = num_unfoldings

        # Will be changed later by the evaluation script for each copy of this dataset
        self.step_to_eval = 1
        self.transition = False

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):

                # max_arb_len = model_kwargs['y']['lengths'].max() #+ 2 * args.handshake_size
                # min_arb_len = model_kwargs['y']['lengths'].min()
                #
                # # assert (min_arb_len > 2 * args.blend_len)
                #
                # for ii, len_s in enumerate(model_kwargs['y']['lengths']):
                #     # model_kwargs['y']['lengths'][ii] += 2 * args.handshake_size
                #     if args.dataset == 'humanml':
                #         model_kwargs['y']['lengths'][ii] = min(
                #             model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 196)
                #     if args.dataset =='babel':
                #         model_kwargs['y']['lengths'][ii] = min(
                #             model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 250)

                    # model_kwargs['y']['lengths'][ii] = min(model_kwargs['y']['lengths'][ii] + 2 * args.handshake_size, 196 if args.dataset == 'humanml' else 250)
                    
                # model_kwargs['y']['lengths'][0] -= args.handshake_size #first and last.
                # model_kwargs['y']['lengths'][-1] -= args.handshake_size


                # Old version:
                max_arb_len = model_kwargs['y']['lengths'].max()
                min_arb_len = 2 * args.handshake_size + 2 * args.blend_len + 10

                for ii, len_s in enumerate(model_kwargs['y']['lengths']):
                    if len_s > max_arb_len:
                        model_kwargs['y']['lengths'][ii] = max_arb_len
                    if len_s < min_arb_len:
                        model_kwargs['y']['lengths'][ii] = min_arb_len
                max_arb_len = model_kwargs['y']['lengths'].max() #+ 2 * args.handshake_size

                n_frames = max_arb_len

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                # tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):

                    sample = []
                    all_texts = []
                    all_lengths = []
                    all_tokens = []

                    batch_size = len(model_kwargs['y']['text'])

                    if scale != 1.:
                        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * scale
                    samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs,
                                                                     n_frames=n_frames, eval_mode=True)
                    # if args.double_take:
                    #     all_samples = samples_per_rep_list[1]  # we only do one rep
                    # else:
                    #     all_samples = samples_per_rep_list[0]  # we only do one rep
                    all_samples = samples_per_rep_list[0]  # we only do one rep
                    sample = all_samples
                    step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
                    for ii, len_i in enumerate(model_kwargs['y']['lengths']):
                        if ii == 0:
                            step_sizes[ii] = len_i
                            continue
                        step_sizes[ii] = step_sizes[ii - 1] + len_i - args.handshake_size
                    final_n_frames = step_sizes[-1]
                    unfolded = unfold_sample_arb_len(sample, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

                    tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                    max_motion_length = int(model_kwargs['y']['lengths'].max())

                    if t == 0:
                        if args.eval_on == "motion":
                            sub_dicts = [{
                                        # eval on seq
                                        'motion': pad_sample_with_zeros(unfolded[..., step_sizes[bs_i] - model_kwargs['y']['lengths'][bs_i] + args.handshake_size:
                                                                step_sizes[bs_i] - args.handshake_size].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                        'length': model_kwargs['y']['lengths'][bs_i] - 2*args.handshake_size,
                                        'caption': model_kwargs['y']['text'][bs_i],
                                        'tokens': tokens[bs_i],
                                        'cap_len': len(tokens[bs_i]),
                                        'is_transition': model_kwargs['y']['is_transition'][bs_i][:args.handshake_size]
                                        } for bs_i in range(1, dataloader.batch_size - 1)] #-1)] uncomment the -1 for transitions
                            sub_dicts += [{
                                'motion': pad_sample_with_zeros(unfolded[..., :step_sizes[0] - args.handshake_size].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                'length': model_kwargs['y']['lengths'][0] - args.handshake_size,
                                'caption': model_kwargs['y']['text'][0],
                                'tokens': tokens[0],
                                'cap_len': len(tokens[0]),
                                'is_transition': model_kwargs['y']['is_transition'][0][:args.handshake_size]
                            }]
                            sub_dicts += [{
                                'motion': pad_sample_with_zeros(unfolded[..., step_sizes[-1] - model_kwargs['y']['lengths'][-1] + args.handshake_size:
                                                                              ].squeeze().permute(1, 0).cpu().numpy(), max_motion_length),
                                'length': model_kwargs['y']['lengths'][-1] - args.handshake_size,
                                'caption': model_kwargs['y']['text'][-1],
                                'tokens': tokens[-1],
                                'cap_len': len(tokens[-1]),
                                'is_transition': model_kwargs['y']['is_transition'][-1][:args.handshake_size]
                            }]
                        elif args.eval_on == "transition":
                            sub_dicts = [{
                                        'motion': unfolded[..., step_sizes[bs_i]-args.handshake_size-(args.transition_margins//2):
                                                                step_sizes[bs_i]+(args.transition_margins//2)].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': args.handshake_size + args.transition_margins,
                                        'caption': model_kwargs['y']['text'][bs_i],
                                        'tokens': tokens[bs_i],
                                        'cap_len': len(tokens[bs_i]),
                                        'is_transition': model_kwargs['y']['is_transition'][bs_i][:args.handshake_size]
                                        } for bs_i in range(0, dataloader.batch_size - 1)] #uncomment the -1 for transitions
                        else:
                            print("Error")
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i],
                                        'length': all_lengths[bs_i],
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': all_texts[bs_i],
                                    'tokens': all_tokens[bs_i],
                                    'cap_len': [len(e) for e in all_tokens[bs_i]],
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, transitions = data['motion'], data['length'], data['caption'], data['tokens'], data['is_transition']
        sent_len = data['cap_len']

        if self.dataset.load_mode == 'eval' and self.dataset.dataset_name != 'babel':  # re-norm is not needed for babel dataset
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []#np.zeros(1)

