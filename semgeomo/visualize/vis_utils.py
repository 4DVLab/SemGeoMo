import numpy as np
from trimesh import Trimesh
import trimesh
import os
import torch
from scipy.spatial.transform import Rotation
from os.path import join as pjoin
from simplify_loc2rot import joints2smpl
from ..model.rotation2xyz import Rotation2xyz


joint_dir  = "/storage/group/4dvlab/congpsh/HHOI/OMOMO/new_joints"

class npy2obj:
    def __init__(self, npy_path, name,device=0, cuda=True):
        if npy_path != None:  #generate
          print("--generate--")
          self.npy_path = npy_path
          self.motions = np.load(self.npy_path, allow_pickle=True)
          if self.npy_path.endswith('.npz'):
              self.motions = self.motions['arr_0']
          self.motions = self.motions[None][0]
          #self.motions['motion'] = self.motions['motion'].reshape(-1,22,3)
        else:
          joint = np.load(pjoin(joint_dir, name + '.npy'))  #gt
          joint = joint[::4] 
          self.motions = dict()
          self.motions['motion'] = joint.reshape(-1,22,3)
        self.nframes,self.njoints, self.nfeats = self.motions['motion'].shape
        self.opt_cache = {}
        self.num_frames = self.motions['motion'].shape[0]
        #self.num_frames = 30
        print(self.num_frames)
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        
        self.name = name
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces

        if self.nfeats == 3:
            print(f'Running SMPLify, it may take a few minutes.')
            motion=self.motions['motion'] #nframex22x3
            motion_tensor,opt_dict = self.j2s.joint2smpl(motion)  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        #self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        #self.real_num_frames = self.motions['lengths'][self.absl_idx]
        self.real_num_frames=self.nframes ###

        #print(self.motions['motion'].shape) #1,25,6,frames
        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True,name=name)
        #print(self.vertices.shape) #1,6890,3,frames
        self.beta=opt_dict['betas']
        #print(self.beta.shape) ###Tx10

    def get_vertices(self, frame_i):
        print(frame_i)
        print(self.vertices[0,:,:,frame_i].shape)
        return self.vertices[0,:, :, frame_i].squeeze().tolist()

    def get_trimesh(self, frame_i):
        return Trimesh(vertices=self.get_vertices(frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0,:, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0,:-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0,-1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0,:, :, :self.real_num_frames],
            #'text': self.motions['text'][0],
            'length': self.real_num_frames,
            'betas':self.beta
        }
        np.save(save_path, data_dict)
        
        

simplified_mesh = {
    "baseball":"baseball/baseball_simplified_transformed.obj",
    'broom':"broom/broom_simplified_transformed.obj",
    'chair':"chair/chair_simplified_transformed.obj",
    'dumbbell':"dumbbell/dumbbell_simplified_transformed.obj",
    'golf':"golf/golf_simplified_transformed.obj",
    'kettlebell':"kettlebell/kettlebell_simplified_transformed.obj",
    'pan':"pan/pan_simplified_transformed.obj",
    'skateboard': "skateboard/skateboard_simplified_transformed.obj",
    'suitcase': "suitcase/suitcase_simplified_transformed.obj",
    'tennis': "tennis/tennis_simplified_transformed.obj",
}    

class npy2obj_object:  #for IMHOI_object
    def __init__(self, npy_path, obj_path, sample_idx, rep_idx, device=0, cuda=True, if_color=False):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.bs, _, self.nfeats, self.nframes = self.motions['motion_obj'].shape

        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        #self.num_frames = self.motions['motion_obj'][self.absl_idx].shape[-1]
        self.num_frames=2

        '''if len( self.motions['obj_name'][0].split('_'))>2:
            obj_name = [b.split('_')[2] for b in self.motions['obj_name']]
        else:
            obj_name = [b for b in self.motions['obj_name']]'''
        #self.motions['motion_obj'] = self.motions['motion_obj'][:,:,:,330:332]
        print(self.motions['motion_obj'].shape)
        obj_name="baseball"
        self.vertices, self.faces = self.pose2mesh(self.motions['motion_obj'], obj_path, obj_name)
        # self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
    
        self.real_num_frames=2


    def pose2mesh(self, motion_obj, obj_path, obj_name):
        vertices_list = []
        faces_list = []
        for b in range(self.bs):
            mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])
            temp_simp = trimesh.load(mesh_path)
            # vertices = temp_simp.vertices * 0.16
            vertices = temp_simp.vertices
            theta = -np.pi / 2
            rotation_matrix = np.array([
                [np.cos(theta), np.sin(theta),0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1] #x>>>y
            ])
            vertices=vertices@rotation_matrix.T
            faces = temp_simp.faces
            # center the meshes
            center = np.mean(vertices, 0)
            vertices -= center
            # transform
            angle, trans = motion_obj[b, 0, :3], motion_obj[b, 0, 3:]
            rot = Rotation.from_rotvec(angle.transpose(1, 0)).as_matrix()
            vertices = np.matmul(vertices[np.newaxis], rot.transpose(0, 2, 1)[:, np.newaxis])[:, 0] + trans.transpose(1, 0)[:, np.newaxis]
            vertices = vertices.transpose(1, 2, 0) 
            vertices_list.append(vertices)
            faces_list.append(faces)
        # return np.stack(vertices_list), np.stack(faces_list)
        return vertices_list, faces_list   #  support any batch_size


    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i][:, :, frame_i].squeeze().tolist()

    def get_faces(self, sample_i):
        return self.faces[sample_i][ :, :].squeeze().tolist()
    
    def get_colors(self, sample_i, frame_i):
        if self.colors is None:
            return None
        else:
            return self.colors[sample_i, ..., frame_i].tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.get_faces(sample_i))

    def save_obj(self, save_path, sample_i, frame_i):
        mesh = self.get_trimesh(sample_i, frame_i) 
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_ply(self, save_path, sample_i, frame_i):
        mesh = self.get_trimesh(sample_i, frame_i)
        mesh.export(save_path)
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion_obj'][self.absl_idx, :, :, :self.real_num_frames],
            'faces': np.array(self.faces[self.absl_idx]),
            'vertices': np.array(self.vertices[self.absl_idx][:, :, :self.real_num_frames]),
            # 'contact_idx': self.contact_idxs[0, :, :self.real_num_frames],
            'contact_idx': None,
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)

