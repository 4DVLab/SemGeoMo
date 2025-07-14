import argparse
import os
import shutil
from tqdm import tqdm
import json
from os.path import join as pjoin
import numpy as np
from os.path import join as pjoin
from plyfile import PlyData, PlyElement
from pywavefront import Wavefront
import vis_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()
    
    name_list = ["sub16_clothesstand_009","sub16_plasticbox_009","sub16_trashcan_000"] #example
    
    root_dir = "../../exp/omomo-pipeline/vis_gif/results/"
    vis_dir = "../../exp/omomo-pipeline/vis_gif/vis/"
    files = os.listdir(root_dir)
    for name in name_list:
        count = 0
        
        npy_path=pjoin(root_dir,name+"scale_results.npy") #ours
        
        print(npy_path)
       
        p = os.path.join("/storage/group/4dvlab/congpsh/HHOI/OMOMO/pc",name+".npy")
        pointcloud =  np.load(p)
        verts=np.array(pointcloud)
        verts = verts[::4]
        joint_p = os.path.join("/storage/group/4dvlab/congpsh/HHOI/OMOMO/new_joints",name+".npy")
        joint =  np.load(joint_p)
        joint=np.array(joint)
        joint = joint[::4]
        
        print(verts.shape)
        for frame in range(0,verts.shape[0]):
            vertices = np.asarray(verts[frame])
            vertex_element = np.zeros(vertices.shape[0], dtype=[
                    ('x', 'f4'),
                    ('y', 'f4'),
                    ('z', 'f4')
                ])
            
            vertex_element['x'] = vertices[:, 0]
            vertex_element['y'] = vertices[:, 1]
            vertex_element['z'] = vertices[:, 2]
           
            file_path=os.path.join(vis_dir,name,"pc")
            os.makedirs(file_path,exist_ok=True)
            ply_file_path=os.path.join(file_path,name+"_"+str(frame)+".ply")
            ply_data = PlyData([PlyElement.describe(vertex_element, 'vertex')], text=True)
            with open(ply_file_path, 'wb') as ply_file:
                ply_data.write(ply_file)
        
        if not os.path.exists(npy_path):
            continue
            
        count += 1
        out_npy_path =os.path.join(vis_dir,name, 'smpl_params.npy')
        assert os.path.exists(npy_path)
        results_dir = os.path.join(vis_dir,name)
        
        if not os.path.exists(results_dir):
          os.makedirs(results_dir)
          
        npy2obj = vis_utils.npy2obj(npy_path,name,
                                    device=params.device, cuda=params.cuda)
                                    
        print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
        npy2obj.save_npy(out_npy_path)
        print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
        for frame_i in tqdm(range(npy2obj.real_num_frames)):  #######
            npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)
            
            
            

    
