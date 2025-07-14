# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import sys
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]
import os
import numpy as np
import pickle
from .smpl import SMPL, JOINTSTYPE_ROOT
import sys
sys.path.append("..")
from utils import rotation_conversions as geometry
from data_loaders.amass.transforms.rots2joints import SMPLH
from data_loaders.humanml.common.skeleton import Skeleton

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
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
example_id = "010225"
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22  
# data_dir = '/remote-home/share/joints/'
# path="/public/home/wangzy17/omomo_dataset/joints/sub10_clothesstand_000.pkl"  #example
# with open(path, 'rb') as f:
#   example_data = pickle.load(f)
data_dir = "/inspurfs/group/mayuexin/congpsh/uni-HOI/process_code/common/"
example_data = np.load(os.path.join(data_dir, example_id + '.npy'))
print("===",os.path.join(data_dir, example_id + '.npy'))
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# print(tgt_skel.shape)
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])


def uniform_skeleton(positions,target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    # print(new_joints[0]-positions[0])
    # print(new_joints[0].shape)
    return new_joints

class Rotation2xyz:
    def __init__(self, device, dataset='amass', batch_size=None):
        self.device = device
        self.dataset = dataset
        if dataset == 'babel':
            self.smpl_model = SMPLH(path='./body_models/smpl_models/smplh',
                                                  jointstype='smplnh',
                                                  input_pose_rep='matrix',
                                                  batch_size=batch_size,
                                                  gender='male',
                                                  name='SMPLH').eval().to(device)

        else:
            self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, name,betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, data_type=None, **kwargs):

        if self.dataset == 'babel':
            out = self.smpl_model(smpl_data=x, batch_size=1)
            return out
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        #print(x.shape)
        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            
        if self.dataset == 'babel':
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas, input_pose_rep='rot6d')
        else:
            print(rotations.shape)
            out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()
        

        floor_height = 0
        if name!=None:
            print(name)
            theta = -np.pi / 2
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]    #z>>>y
            ])
            jointp=os.path.join("/storage/group/4dvlab/congpsh/HHOI/OMOMO/joints/",name+".pkl")
            with open(jointp, 'rb') as f:
                joint= pickle.load(f)
            joint = np.array(joint[:,:22,:])
            newjoint = joint - joint[0,0] #nframesx22x3  motion
            rotated_joint=newjoint[:,:22]@ rotation_matrix.T 
            old_joint = rotated_joint.copy()
            uniform_joint = uniform_skeleton(old_joint,tgt_offsets)
            floor_height = uniform_joint.min(axis=0).min(axis=0)[1]
        #print("---------")
        print(floor_height)

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]
            for i in range(x_xyz.shape[0]):
                x_xyz[i,:,1,:]-=floor_height

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
