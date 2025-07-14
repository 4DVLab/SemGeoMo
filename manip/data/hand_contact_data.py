import os
import numpy as np
import joblib 
import rich
import pickle
import random
import json 
import trimesh
import codecs as cs 
import time 
from manip.data.word_vectorizer import WordVectorizer
import codecs as cs
from bps_torch.bps import bps_torch
from os.path import join as pjoin
from rich.progress import track
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import chardet

import pytorch3d.transforms as transforms 

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder

from manip.lafan1.utils import rotate_at_frame_w_obj 

from semgeomo.data_loaders.humanml.common.quaternion import *
from semgeomo.data_loaders.humanml.common.skeleton import Skeleton

t2m_raw_offsets = np.array([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
example_id = "010225"
l_idx1, l_idx2 = 5, 8
fid_r, fid_l = [8, 11], [7, 10]
face_joint_indx = [2, 1, 17, 16]
r_hip, l_hip = 2, 1
joints_num = 22  
data_dir = "/inspurfs/group/mayuexin/congpsh/uni-HOI/process_code/common/"
example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
def process_joint(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    
    global_positions = positions.copy()
    
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)


    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(dataset.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data
    

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

SMPLH_PATH = "/storage/group/4dvlab/congpsh/HHOI/OMOMO/smplh"
DATASETS = {
    'omomo': [1,"/storage/group/4dvlab/congpsh/HHOI/OMOMO/",100],
    'behave': [2,"/storage/group/4dvlab/congpsh/HHOI/behave_t2m/",200]
}

def remove_null_bytes(input_file_path, output_file_path):
    with open(input_file_path, 'rb') as infile:
        content = infile.read()
        # Remove null bytes from content
        cleaned_content = content.replace(b'\x00', b'')
    
    with open(output_file_path, 'wb') as outfile:
        outfile.write(cleaned_content)
        
        
class HandContactDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        dataset_name='omomo',
        window=120,
        use_object_splits=False,
        tmpFile=True,
        debug = False,
    ):
        print(dataset_name)
        if dataset_name == "omomo":
            self.max_motion_length = 100 ###omomo:fps15-100 / behave:200
        else:
            self.max_motion_length = 200
        self.train = train
        self.window = window
        self.dataset_name = dataset_name
        self.bps_path = "./manip/data/bps.pt"
        self.bps = torch.load(self.bps_path)
        self.prep_bps_data()
        self.obj_bps = self.bps['obj'] 
        self.bps_torch = bps_torch()
        self.unit_length = 4
        
        self.contact = True
        
        
        if self.train:
            split = 'train'
        else:
            split = 'test'
            
        new_name_list = []
        length_list = []
        motion_dict = {}

        if dataset_name == "omomo":
            self.data_fps = "/storage/group/4dvlab/wangzy/SemGeoMo/data_pkl/omomo_fps15/"

            self.data_root = DATASETS[dataset_name][1]
            split_file = pjoin(self.data_root, split + '.txt')
            motion_dir = pjoin(self.data_root, 'new_joint_vecs_fps15') #fps15
            joint_dir = pjoin(self.data_root, 'new_joints')
            pc_dir = pjoin(self.data_root, 'pc')
            text_dir = pjoin(self.data_root, 'texts')
            RT_dir = pjoin(self.data_root, 'newRT')
            fine_text_dir = text_dir.replace('texts','fine_text4')

            self.obj_geo_root_folder = "/storage/group/4dvlab/congpsh/HHOI/OMOMO/captured_objects/"
        
            self.mean = np.load(pjoin(self.data_fps,"mean.npy"))
            self.std = np.load(pjoin(self.data_fps,"std.npy"))
            
            self.id_list = []  
            print(split_file)   
            with cs.open(split_file, "r",encoding='utf-8',errors='ignore') as f:
                for line in f.readlines():
                    self.id_list.append(str(line.strip()))
                    
            selected_list = random.sample(self.id_list, len(self.id_list)//5)
                    
            # enumerator = enumerate(self.id_list)
            if not debug:
                enumerator = enumerate(self.id_list)
                maxdata = 1e10
                subset = ''
            else:
                print(f"--Loading from {self.data_root} {split} debug")
                maxdata = 50
                enumerator = enumerate(self.id_list[:maxdata])
                subset = '_tiny'

        elif dataset_name == "behave":
            subset = ''
            self.data_fps = "/storage/group/4dvlab/wangzy/SemGeoMo/data_pkl/behave/"
            self.mean = np.load(pjoin(self.data_fps,"mean.npy"))
            self.std = np.load(pjoin(self.data_fps,"std.npy"))

        # Fast loading
        if os.path.exists(pjoin(self.data_fps, f'tmp/{split}_hoi_motion.pkl')):
            with rich.progress.open(pjoin(self.data_fps, f'tmp/{split}_hoi_motion.pkl'),
                    'rb', description=f"Loading {self.data_fps} {split}") as file:
                motion_dict = pickle.load(file)
            with open(pjoin(self.data_fps, f'tmp/{split}_hoi_index.pkl'), 'rb') as file:
                new_name_list = pickle.load(file)
            # print(len(motion_dict),motion_dict[new_name_list[0]].keys())
        else:
            # print("length of the dataset:",len(new_name_list))
            # print(self.id_list[1])
            print("Slow loading dataset!")
            
            if dataset_name == "omomo":
                for idx, name in tqdm(enumerator):

                    name = name.replace('\x00', '')
                    if name == '':
                        print("none")
                        continue
                        
                    if not os.path.exists(f"{motion_dir}/{name}.npy"):
                        print(f"{motion_dir}/{name}.npy is not exist!")
                        continue
                    if not os.path.exists(f"{joint_dir}/{name}.npy"):
                        print(f"{joint_dir}/{name}.npy is not exist!")
                        continue
                    
                    motion = np.load(f"{motion_dir}/{name}.npy",allow_pickle=True) #fps15 (T-1)
                    joint = np.load(f"{joint_dir}/{name}.npy",allow_pickle=True)
                    
                    norm_contact_dis = np.zeros((joint.shape[0],3,2))
                    if self.contact:
                        contact_dis_path = pjoin(self.data_root, 'contact_dist',name + ".npy") 
                        if not os.path.exists(contact_dis_path):
                            print(f"{contact_dis_path} is not exist!")
                            continue
                        contact_dis = np.load(contact_dis_path)# T X N x 2
                        
                        norm_contact_dis = self.z_score_normalize(contact_dis)
                        
                    if len(joint.shape) == 4:
                        joint = joint[0] 

                    pc = np.load(f"{pc_dir}/{name}.npy",allow_pickle=True)
                    RT = np.load(f"{RT_dir}/{name}.npy",allow_pickle=True)
                    scale = np.load(pjoin("/storage/group/4dvlab/congpsh/HHOI/OMOMO/objscale/",name+".npy"))
                    RT = RT[::4][:-1]
                    scale = scale[::4][:-1]
                    norm_contact_dis  = norm_contact_dis[::4][:-1]
                    pc = pc[::4][:-1]
                    joint = joint[::4][:-1]

                    self.bps_root = "/storage/group/4dvlab/wangzy/SemGeoMo/bps/omomo-bps"
                    bps_p = pjoin(self.bps_root, name + ".npy")  ##
                    if os.path.exists(bps_p):
                        pc_bps = np.load(bps_p)
                    else:
                        pc_bps = self.compute_object_geo_bps(pc)#tensor
                        pc_bps = pc_bps.cpu()
                        np.save(bps_p,np.array(pc_bps))

                    pc_bps = pc_bps[:-1]

                    rot = RT[:,:3]
                    trans = RT[:,3:]
                    
                    # Read text
                    text_data = []
                    if not os.path.exists(f"{text_dir}/{name}.txt") or not os.path.exists(f"{fine_text_dir}/{name}.txt"):
                        print( f"{text_dir}/{name}.txt!" or f"{fine_text_dir}/{name}.txt" "not exists!")
                        continue
                    
                    with cs.open(pjoin(fine_text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption_fine = line_split[0]

                    with cs.open(f"{text_dir}/{name}.txt") as f:
                        lines = f.readlines()
                        for line in lines:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            
                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            
                            text_data.append(text_dict)
                            
                            # print("=====",name)
                    if DATASETS[dataset_name][0] != -1:
                        obj_name = name.split('_')[DATASETS[dataset_name][0]]
                    else:
                        obj_name = name
                        
                    trans = trans.reshape(-1,3) ####object trans Tx3
                    rot_mat= np.array(axis_angle_to_matrix(torch.from_numpy(rot))) #TX3X3

                    print(motion.shape)
                    print(joint.shape)
                    print(pc.shape)
                    print(pc_bps.shape)
                    print(scale.shape)
                    print(trans.shape)
                    print(rot_mat.shape)
                    print("-----------------")
                    
                
                    motion_dict[name] = {
                        "motion": np.array(motion,dtype=np.float32),
                        "length": len(motion),
                        "text": text_data,
                        "fine_text":caption_fine,
                        "joint": np.array(joint,dtype=np.float32),
                        "seq_name": name,
                        "id": idx,  
                        "obj_name": obj_name,
                        "pc": np.array(pc,dtype=np.float32),
                        "dis":np.array(norm_contact_dis,dtype=np.float32),
                        "pc_bps":np.array(pc_bps,dtype=np.float32),
                        "obj_rot_mat":np.array(rot_mat,dtype=np.float32),
                        "obj_trans": np.array(trans,dtype=np.float32),
                        "obj_scale": np.array(scale,dtype=np.float32),
                    }
                    new_name_list.append(name)
                    # except:
                    #     pass
                # if self.dataset_name != 'behave' and self.dataset_name != 'omomo':
                #     tmpFile = False

            elif dataset_name == "behave":
                print("behave--------------")
                data_root = "/storage/group/4dvlab/congpsh/HHOI/behave_t2m/"
                data_dir = "/storage/group/4dvlab/congpsh/HHOI/behave_t2m/new_joint_vecs/"
                joint_dir =  "/storage/group/4dvlab/congpsh/HHOI/behave_t2m/new_joints/"
                pc_dir = pjoin(data_root,"pc")
                text_dir = pjoin(data_root,"pred_text")
                split_file = pjoin(data_root,split+".txt")
                id_list = []
                
                with open(split_file, 'rb') as f:
                    result = chardet.detect(f.read())
                    encoding = result['encoding']
                    
                with open(split_file, 'r', encoding=encoding) as f:
                    for line in f.readlines():
                        id_list.append(line.strip())
                
                print(len(id_list))
                
                for name in tqdm(id_list):
                    
                    if not os.path.exists(pjoin(text_dir, name + '.txt')):
                        print("no text")
                        continue
                    if not os.path.exists(pjoin(data_dir, name + '.npy')):
                        print("no motion")
                        continue

                    motion = np.load(pjoin(data_dir, name + '.npy')) #T-1
                    joint = np.load(pjoin(joint_dir, name + '.npy'))
                    
                    if np.isnan(motion).any():
                        print("nan!")
                        print(len(motion))
                        continue
                    
                    if (len(motion)) < 5 or (len(motion) > 200):
                        print(motion.shape)
                        print("too short/long")
                        continue
                      
                    norm_contact_dis = np.zeros((joint.shape[0],3,2))
                        # just for omomo training
                    if self.contact:
                        contact_dis_path = pjoin(data_root, 'contact_dist',name + ".npy") 
                        if not os.path.exists(contact_dis_path):
                            print(f"{contact_dis_path} is not exist!")
                            continue
                        contact_dis = np.load(contact_dis_path)# T X N x 2
                        norm_contact_dis = self.z_score_normalize(contact_dis)
                  
                    
                    if len(joint.shape) == 4:
                        joint = joint[0] 
                        
                    pc = np.load(f"{pc_dir}/{name}.npy",allow_pickle=True)
                    
                    self.bps_root = "/storage/group/4dvlab/wangzy/uni_regen/bps/behave-bps/" #15
                    bps_p = pjoin(self.bps_root, name + ".npy")  ##
                    if os.path.exists(bps_p):
                        pc_bps = np.load(bps_p)
                    else:
                        pc_bps = self.compute_object_geo_bps(pc)#tensor
                        pc_bps = pc_bps.cpu()
                        np.save(bps_p,np.array(pc_bps))
                    
                    
                    # Read text
                    text_data = []
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
                            
                    
                            text_data.append(text_dict)
                            
                    
                    frames = min(joint.shape[0], pc.shape[0])
                    joint = joint[:frames]
                    pc = pc[:frames]
                    pc_bps = pc_bps[:frames]
                    motion = motion[:frames]
                    norm_contact_dis = norm_contact_dis[:frames]

                    print(joint.shape)
                    print(pc.shape)
                    print(pc_bps.shape)
                    print(norm_contact_dis.shape)
                    print("-----------------")
                    
                    
                    motion_dict[name] = {
                        "motion": np.array(motion,dtype=np.float32),
                        "length": len(motion),
                        "text": text_data,
                        "joint": np.array(joint,dtype=np.float32),
                        "seq_name": name,
                        "pc": np.array(pc,dtype=np.float32),
                        "dis":np.array(norm_contact_dis,dtype=np.float32),
                        "pc_bps":np.array(pc_bps,dtype=np.float32),
                    }
                    new_name_list.append(name)
                    
            if tmpFile:
                os.makedirs(pjoin(self.data_fps, 'tmp'), exist_ok=True)
                with open(pjoin(self.data_fps, f'tmp/{split}{subset}_hoi_motion.pkl'),'wb') as file:
                    pickle.dump(motion_dict, file)
                with open(pjoin(self.data_fps, f'tmp/{split}{subset}_hoi_index.pkl'), 'wb') as file:
                    pickle.dump(new_name_list, file)

        self.motion_dict = motion_dict
        self.name_list = new_name_list
        print(len(self.motion_dict))
        
        min_max_mean_std_data_path = os.path.join(self.data_fps, "min_max_mean_std_data_window_"+str(self.window)+"_cano_joints24.p")
        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                print("min/max")
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)
            
        self.global_jpos_min = min_max_mean_std_jpos_data['global_jpos_min'].reshape(1,22, 3)
        self.global_jpos_max = min_max_mean_std_jpos_data['global_jpos_max'].reshape(1,22, 3)
          
        
    
    def apply_transformation_to_obj_geometry(self, obj_mesh_path, obj_scale, obj_rot, obj_trans):
        mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh_verts = np.asarray(mesh.vertices) # Nv X 3
        obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3 

        ori_obj_verts = torch.from_numpy(obj_mesh_verts).float()[None].repeat(obj_trans.shape[0], 1, 1) # T X Nv X 3 
    
        seq_scale = torch.from_numpy(obj_scale).float() # T 
        seq_rot_mat = torch.from_numpy(obj_rot).float() # T X 3 X 3 
        if obj_trans.shape[-1] != 1:
            seq_trans = torch.from_numpy(obj_trans).float()[:, :, None] # T X 3 X 1 
        else:
            seq_trans = torch.from_numpy(obj_trans).float() # T X 3 X 1 
        transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
        seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
        transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

        return transformed_obj_verts, obj_mesh_faces  

    def load_object_geometry(self, object_name, obj_scale, obj_trans, obj_rot, \
        obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):
        obj_mesh_path = os.path.join(self.obj_geo_root_folder, object_name+"_cleaned_simplified.obj")
        obj_mesh_verts, obj_mesh_faces =self.apply_transformation_to_obj_geometry(obj_mesh_path, \
        obj_scale, obj_rot, obj_trans) # T X Nv X 3 

        return obj_mesh_verts, obj_mesh_faces 
    
    def to_tensor(self,array, dtype=torch.float32):
        if not torch.is_tensor(array):
            array = torch.tensor(array)
        return array.to(dtype)

    def extract_min_max_mean_std_from_data(self):
        
        all_global_jpos_data,all_global_jvel_data = [],[]
        for s_idx in self.motion_dict.keys():
            joints = self.motion_dict[s_idx]['joint'] # T X D 
            if len(joints.shape) == 4:
                joints = joints[0] 
            all_global_jpos_data.append(joints.reshape(-1,66))
            all_global_jvel_data.append(joints[1:] - joints[:-1])

        all_global_jpos_data = np.vstack(all_global_jpos_data).reshape(-1, 66) # (N*T) X 72 
        all_global_jvel_data = np.vstack(all_global_jvel_data).reshape(-1, 66)

        min_jpos = all_global_jpos_data.min(axis=0)
        max_jpos = all_global_jpos_data.max(axis=0)
        min_jvel = all_global_jvel_data.min(axis=0)
        max_jvel = all_global_jvel_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_jpos_min'] = min_jpos 
        stats_dict['global_jpos_max'] = max_jpos 
        stats_dict['global_jvel_min'] = min_jvel 
        stats_dict['global_jvel_max'] = max_jvel  

        return stats_dict 
        
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
    
    def z_score_normalize(self,data):
        mean = np.mean(data, axis=(0, 1), keepdims=True)
        std = np.std(data, axis=(0, 1), keepdims=True)
        normalized_data = (data - mean) / std
        return normalized_data

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            
            bps = {
                'obj': bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj'] 
        
    def process_joint(self,positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    
        global_positions = positions.copy()
        
        """ Get Foot Contacts """
    
        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
    
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            #     feet_l_h = positions[:-1,fid_l,1]
            #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float)
    
            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            #     feet_r_h = positions[:-1,fid_r,1]
            #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
            return feet_l, feet_r
    
        #
        feet_l, feet_r = foot_detect(positions, feet_thre)
        # feet_l, feet_r = foot_detect(positions, 0.002)
    
        '''Quaternion and Cartesian representation'''
        r_rot = None
    
        def get_rifke(positions):
            '''Local pose'''
            positions[..., 0] -= positions[:, 0:1, 0]
            positions[..., 2] -= positions[:, 0:1, 2]
            '''All pose face Z+'''
            positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
            return positions
    
        def get_quaternion(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)
    
            '''Fix Quaternion Discontinuity'''
            quat_params = qfix(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            quat_params[1:, 0] = r_velocity
            # (seq_len, joints_num, 4)
            return quat_params, r_velocity, velocity, r_rot
    
        def get_cont6d_params(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    
            '''Quaternion to continuous 6D'''
            cont_6d_params = quaternion_to_cont6d_np(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
            # (seq_len, joints_num, 4)
            return cont_6d_params, r_velocity, velocity, r_rot
    
        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
        positions = get_rifke(positions)
    
    
        '''Root height'''
        root_y = positions[:, 0, 1:2]
    
        '''Root rotation and linear velocity'''
        # (seq_len-1, 1) rotation velocity along y-axis
        # (seq_len-1, 2) linear velovity on xz plane
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    
        '''Get Joint Rotation Representation'''
        # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    
        '''Get Joint Rotation Invariant Position Represention'''
        # (seq_len, (joints_num-1)*3) local joint position
        ric_data = positions[:, 1:].reshape(len(positions), -1)
    
        '''Get Joint Velocity Representation'''
        # (seq_len-1, joints_num*3)
        local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                            global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)
    
        data = root_data
        data = np.concatenate([data, ric_data[:-1]], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        #     print(dataset.shape, local_vel.shape)
        data = np.concatenate([data, local_vel], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)
    
        return data


        
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index): ####
        # index = 0 # For debug 
        # print(len(self.motion_dict))
        data = self.motion_dict[self.name_list[index]]
        # print(len(self.data_dict))
        
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        joint = data["joint"]
        
        if joint.shape[0]<=1:  #skip this
          index += 1
          data = self.motion_dict[self.name_list[index]]
          motion, m_length, text_list = data["motion"], data["length"], data["text"]
        
        if self.dataset_name =="omomo":
        
            rot_mat,trans,obj_scale = data["obj_rot_mat"], data["obj_trans"], data["obj_scale"]
                
            motion,joint, pc = data["motion"], data["joint"],data["pc"]
            fine_text = data["fine_text"]
            
            "Z Normalization"
            motion = (motion - self.mean) / self.std

            # joint = joint[:-1]
            # pc = pc[:-1]
            
            #print(joint.shape,pc.shape,motion.shape)
            norm_contact_dis = np.zeros((data["joint"].shape[0],3,2))
            if self.contact:
                norm_contact_dis = data["dis"]
                
            num_joints = 22
            normalized_jpos = self.normalize_jpos_min_max(joint[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
            pc_bps = data["pc_bps"]
            min_length = min(joint.shape[0],pc_bps.shape[0],pc.shape[0],motion.shape[0],norm_contact_dis.shape[0])
            m_length = min_length
            norm_contact_dis = norm_contact_dis[:min_length]
            pc_bps = pc_bps[:min_length]
            normalized_jpos = normalized_jpos[:min_length]
            joint = joint[:min_length]
            pc = pc[:min_length]
            motion = motion[:min_length]
            rot_mat = rot_mat[:min_length]
            trans = trans[:min_length]
            obj_scale = obj_scale[:min_length]
            
            if len(joint.shape) == 4:
                joint = joint[0]
                
            # print(joint_root.shape)
                
            if m_length < self.max_motion_length:
                motion = np.concatenate([motion,np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
                normalized_jpos = np.concatenate([normalized_jpos,np.zeros((self.max_motion_length - m_length, normalized_jpos.shape[1],3))], axis=0)
                pc = np.concatenate([pc,np.zeros((self.max_motion_length - m_length, pc.shape[1],pc.shape[2]))], axis=0)
                rot_mat = np.concatenate([rot_mat,np.zeros((self.max_motion_length - m_length, rot_mat.shape[1],rot_mat.shape[2]))], axis=0)
                trans = np.concatenate([trans,np.zeros((self.max_motion_length - m_length, trans.shape[1]))], axis=0)
                obj_scale = np.concatenate([obj_scale,np.zeros((self.max_motion_length - m_length))], axis=0)
                pc_bps = np.concatenate([pc_bps,np.zeros((self.max_motion_length - m_length, pc_bps.shape[1],pc_bps.shape[2]))], axis=0)
                joint = np.concatenate([joint,np.zeros((self.max_motion_length - m_length, joint.shape[1],3))], axis=0)
                norm_contact_dis = np.concatenate([norm_contact_dis,np.zeros((self.max_motion_length - m_length, norm_contact_dis.shape[1],2))], axis=0)
            else:
                # print(len(motion),self.max_motion_length)
                idx = random.randint(0, len(motion) - self.max_motion_length)
                motion = motion[idx:idx + self.max_motion_length]                       # T x 263
                normalized_jpos = normalized_jpos[idx:idx + self.max_motion_length]     # T x 22 x 3
                pc = pc[idx:idx + self.max_motion_length]                               # T x 1024 x 3
                rot_mat = rot_mat[idx:idx + self.max_motion_length]          
                trans = trans[idx:idx + self.max_motion_length]          
                obj_scale = obj_scale[idx:idx + self.max_motion_length]          
                pc_bps = pc_bps[idx:idx + self.max_motion_length]                       # T x 1024 x 3
                joint = joint[idx:idx + self.max_motion_length]                         # T x 22 x 3
                norm_contact_dis = norm_contact_dis[idx:idx + self.max_motion_length]   # T x 1024 x 2
                
            # print(pc_bps.shape,norm_contact_dis.shape)
            # print(motion.shape,pc.shape,normalized_jpos.shape,pc_bps.shape,norm_contact_dis.shape,joint.shape)
            # T x 3

            # Randomly select a caption
            text_data = random.choice(text_list)
            #print(text_data)
            caption, tokens = text_data["caption"], text_data["tokens"]
            
            
            data_input_dict = {
                "motion": motion,   #nx263
                "seq_len": m_length,
                "text": caption,
                "fine_text":fine_text,
                "joint": normalized_jpos, #nx22x3
                "seq_name": data["seq_name"],
                "obj_bps":pc_bps,  #pc_bps nx1024x3
                "obj_com_pos": pc.mean(1),
                "pc":pc,#nx1024x3
                "index":data["id"],
                "dis":norm_contact_dis, #nx1024x2
                "obj_rot_mat": rot_mat, #nx3x3
                "obj_trans": trans, #nx3
                "obj_scale": obj_scale #n
            }
        
        elif self.dataset_name == "behave":
          
            joint, pc,motion= data["joint"], data["pc"],data["motion"]

            "Z Normalization"
            motion = (motion - self.mean) / self.std
        
            norm_contact_dis = np.zeros((data["joint"].shape[0],3,2))
            if self.contact:
                norm_contact_dis = data["dis"]
                
            num_joints = 22
            normalized_jpos = self.normalize_jpos_min_max(joint[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
            pc_bps = data["pc_bps"]
            min_length = min(joint.shape[0],pc_bps.shape[0],pc.shape[0],motion.shape[0],norm_contact_dis.shape[0])
            m_length = min_length
            norm_contact_dis = norm_contact_dis[:min_length]
            pc_bps = pc_bps[:min_length]
            normalized_jpos = normalized_jpos[:min_length]
            joint = joint[:min_length]
            pc = pc[:min_length]
            motion = motion[:min_length]
            
            if len(joint.shape) == 4:
                joint = joint[0]
                
            # print(joint_root.shape)
                
            if m_length < self.max_motion_length:
                motion = np.concatenate([motion,np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0)
                normalized_jpos = np.concatenate([normalized_jpos,np.zeros((self.max_motion_length - m_length, normalized_jpos.shape[1],3))], axis=0)
                pc = np.concatenate([pc,np.zeros((self.max_motion_length - m_length, pc.shape[1],pc.shape[2]))], axis=0)
                pc_bps = np.concatenate([pc_bps,np.zeros((self.max_motion_length - m_length, pc_bps.shape[1],pc_bps.shape[2]))], axis=0)
                joint = np.concatenate([joint,np.zeros((self.max_motion_length - m_length, joint.shape[1],3))], axis=0)
                norm_contact_dis = np.concatenate([norm_contact_dis,np.zeros((self.max_motion_length - m_length, norm_contact_dis.shape[1],2))], axis=0)
            else:
                # print(len(motion),self.max_motion_length)
                idx = random.randint(0, len(motion) - self.max_motion_length)
                motion = motion[idx:idx + self.max_motion_length]                       # T x 263
                normalized_jpos = normalized_jpos[idx:idx + self.max_motion_length]     # T x 22 x 3
                pc = pc[idx:idx + self.max_motion_length]                               # T x 1024 x 3
                pc_bps = pc_bps[idx:idx + self.max_motion_length]                       # T x 1024 x 3
                joint = joint[idx:idx + self.max_motion_length]                         # T x 22 x 3
                norm_contact_dis = norm_contact_dis[idx:idx + self.max_motion_length]   # T x 1024 x 2               # T x 22 x 3
                
            # print(pc_bps.shape,norm_contact_dis.shape)
            # print(motion.shape,pc.shape,normalized_jpos.shape,pc_bps.shape,norm_contact_dis.shape,joint.shape)
            # T x 3

            #print(norm_contact_dis.shape)

            if pc.shape[1] == 512 :  #behave
                pc = np.repeat(pc, 2,1)
            if norm_contact_dis.shape[1] ==512:
                norm_contact_dis = np.repeat(norm_contact_dis, 2,1)


            text_dir = "/storage/group/4dvlab/congpsh/HHOI/behave_t2m/pred_text"
            name = data["seq_name"]
            c_list = []
            with cs.open(f"{text_dir}/{name}.txt") as f:
                lines = f.readlines()
                for line in lines:
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    t_tokens = line_split[1].split(' ')
                    c_list.append(caption)
            
            caption = random.choice(c_list)
            
            data_input_dict = {
                "motion": motion,   #nx263
                "seq_len": m_length,
                "text": caption,
                "joint": normalized_jpos, #nx22x3
                "seq_name": data["seq_name"],
                "pc":pc,#nx1024x3
                "obj_bps":pc_bps,  #pc_bps nx1024x3
                "obj_com_pos": pc.mean(1),
                "dis":norm_contact_dis,
            }
            
            
        return data_input_dict 
        # data_input_dict['motion']: T X (24*3+22*6) range [-1, 1]
        # data_input_dict['obj_bps]: T X N X 3 

    # def __getitem__(self, index): ####