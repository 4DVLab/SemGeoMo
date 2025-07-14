import pickle
import numpy as np 
import os
import numpy as np
import joblib 
import json 
import trimesh 
import time 
from plyfile import PlyData, PlyElement
from pywavefront import Wavefront
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from os.path import join as pjoin
from scipy.optimize import least_squares
import cv2

import torch.nn.functional as F
from torch.autograd import Variable

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
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
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
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def get_smpl_parents(use_joints24=False):
    bm_path = SMPLH_PATH
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents

def apply_transformation_to_obj_geometry(obj_mesh_path, obj_scale, obj_rot, obj_trans):
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
    seq_scale = seq_scale[:-1]
    print(seq_trans.shape)
    print(seq_scale.shape)
    print(seq_rot_mat.shape)
    print(ori_obj_verts.shape)
    transformed_obj_verts = seq_scale.unsqueeze(-1).unsqueeze(-1) * \
    seq_rot_mat.bmm(ori_obj_verts.transpose(1, 2)) + seq_trans
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 
    return transformed_obj_verts, obj_mesh_faces  


def load_object_geometry(obj_geo_root_folder, object_name, obj_scale, obj_trans, obj_rot, \
    obj_bottom_scale=None, obj_bottom_trans=None, obj_bottom_rot=None):

    obj_mesh_path = os.path.join(obj_geo_root_folder, object_name+"_cleaned_simplified.obj")
    print(obj_mesh_path)
    if object_name == "vacuum" or object_name == "mop":
        two_parts = False ##
    else:
        two_parts = False 

    if two_parts:
        print("2 parts")
        top_obj_mesh_path = os.path.join(obj_geo_root_folder, object_name+"_cleaned_simplified_top.obj")
        bottom_obj_mesh_path = os.path.join(obj_geo_root_folder, object_name+"_cleaned_simplified_bottom.obj")

        top_obj_mesh_verts, top_obj_mesh_faces = apply_transformation_to_obj_geometry(top_obj_mesh_path, \
        obj_scale, obj_rot, obj_trans)
        bottom_obj_mesh_verts, bottom_obj_mesh_faces =apply_transformation_to_obj_geometry(bottom_obj_mesh_path, \
        obj_bottom_scale, obj_bottom_rot, obj_bottom_trans)

        obj_mesh_verts, obj_mesh_faces = merge_two_parts([top_obj_mesh_verts, bottom_obj_mesh_verts], \
        [top_obj_mesh_faces, bottom_obj_mesh_faces])
    else:
        print("no 2 parts!")
        obj_mesh_verts, obj_mesh_faces =apply_transformation_to_obj_geometry(obj_mesh_path,obj_scale, obj_rot, obj_trans) # T X Nv X 3 
    return obj_mesh_verts, obj_mesh_faces 


def merge_two_parts(verts_list, faces_list):
    verts_num = 0
    merged_verts_list = []
    merged_faces_list = []
    for p_idx in range(len(verts_list)):
        # part_verts = torch.from_numpy(verts_list[p_idx]) # T X Nv X 3 
        part_verts = verts_list[p_idx] # T X Nv X 3 
        part_faces = torch.from_numpy(faces_list[p_idx]) # T X Nf X 3 

        if p_idx == 0:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces)
        else:
            merged_verts_list.append(part_verts)
            merged_faces_list.append(part_faces+verts_num)

        verts_num += part_verts.shape[1] 

    # merged_verts = torch.cat(merged_verts_list, dim=1).data.cpu().numpy()
    merged_verts = torch.cat(merged_verts_list, dim=1)
    merged_faces = torch.cat(merged_faces_list, dim=0).data.cpu().numpy() 

    return merged_verts, merged_faces 

def estimate_rot_trans(V, objVerts):
    n = V.shape[0]
    assert objVerts.shape == (n, 3)
    A = np.hstack((objVerts, np.ones((n, 1))))
    X, residuals, _, _ = np.linalg.lstsq(A, V, rcond=None)
    rot = X[:3, :3].T  
    trans = X[3,:3]  
    return rot, trans

def transform_template(objVerts, rot, trans):
    trans = trans
    rot = rot
    rot = axis_angle_to_matrix(rot).view(3,3)
    return torch.mm(objVerts, rot.T) + trans
    
  




  
