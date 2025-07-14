import os 
import numpy as np
import time 

import json 


from sklearn.cluster import DBSCAN

import trimesh 

import torch 

import torch.nn.functional as F

def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3 
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)

def determine_floor_height_and_contacts(body_joint_seq):
    '''
    Input: body_joint_seq N x 22 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    FLOOR_VEL_THRESH = 0.005
    FLOOR_HEIGHT_OFFSET = 0.01

    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, 0, :]
    left_toe_seq = body_joint_seq[:, 10, :]
    right_toe_seq = body_joint_seq[:, 11, :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (y is up)
    left_toe_heights = left_toe_seq[:, 1]  # Modify to use y-axis (index 1) for height
    right_toe_heights = right_toe_seq[:, 1]
    root_heights = root_seq[:, 1]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
       
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
           
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
           
            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

    else:
        floor_height = offset_floor_height = 0.0
   
    return floor_height

def compute_foot_sliding_for_smpl(pred_global_jpos, floor_height):  #fps
    # pred_global_jpos: T X J X 3 
    seq_len = pred_global_jpos.shape[0]

    # Put human mesh to floor y = 0 and compute. 
    pred_global_jpos[:, :, 1] -= floor_height  # Modify to use y-axis (index 1) for height

    lankle_pos = pred_global_jpos[:, 7, :] # T X 3 
    ltoe_pos = pred_global_jpos[:, 10, :] # T X 3 

    rankle_pos = pred_global_jpos[:, 8, :] # T X 3 
    rtoe_pos = pred_global_jpos[:, 11, :] # T X 3 

    H_ankle = 0.08 # meter  so for cm should be 100
    H_toe = 0.04 # meter 

    lankle_disp = np.linalg.norm(lankle_pos[1:, ::2] - lankle_pos[:-1, ::2], axis = 1) # T , x and z displacement (0th and 2nd index)
    ltoe_disp = np.linalg.norm(ltoe_pos[1:, ::2] - ltoe_pos[:-1, ::2], axis = 1) # T 
    rankle_disp = np.linalg.norm(rankle_pos[1:, ::2] - rankle_pos[:-1, ::2], axis = 1) # T 
    rtoe_disp = np.linalg.norm(rtoe_pos[1:, ::2] - rtoe_pos[:-1, ::2], axis = 1) # T 

    lankle_subset = lankle_pos[:-1, 1] < H_ankle  # Modify to use y-axis (index 1) for height check
    ltoe_subset = ltoe_pos[:-1, 1] < H_toe
    rankle_subset = rankle_pos[:-1, 1] < H_ankle
    rtoe_subset = rtoe_pos[:-1, 1] < H_toe
   
    lankle_sliding_stats = np.abs(lankle_disp * (2 - 2 ** (lankle_pos[:-1, 1]/H_ankle)))[lankle_subset]
    lankle_sliding = np.sum(lankle_sliding_stats)/seq_len * 100

    ltoe_sliding_stats = np.abs(ltoe_disp * (2 - 2 ** (ltoe_pos[:-1, 1]/H_toe)))[ltoe_subset]
    ltoe_sliding = np.sum(ltoe_sliding_stats)/seq_len * 100

    rankle_sliding_stats = np.abs(rankle_disp * (2 - 2 ** (rankle_pos[:-1, 1]/H_ankle)))[rankle_subset]
    rankle_sliding = np.sum(rankle_sliding_stats)/seq_len * 100

    rtoe_sliding_stats = np.abs(rtoe_disp * (2 - 2 ** (rtoe_pos[:-1, 1]/H_toe)))[rtoe_subset]
    rtoe_sliding = np.sum(rtoe_sliding_stats)/seq_len * 100

    sliding = (lankle_sliding + ltoe_sliding + rankle_sliding + rtoe_sliding) / 4 / 4. 

    return sliding

def compute_s1_metrics(ori_jpos_pred, ori_jpos_gt):
    # pred_hand_jpos: T X 2 X 3
    # gt_hand_jpos: T X 2 X 3 

    ori_jpos_pred = ori_jpos_pred.reshape(-1, 2, 3)
    ori_jpos_gt = ori_jpos_gt.reshape(-1, 2, 3)

    lhand_idx = 0
    rhand_idx = 1
    lhand_jpos_pred = ori_jpos_pred[:, lhand_idx, :].detach().cpu().numpy() 
    rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpos_gt = ori_jpos_gt[:, lhand_idx, :].detach().cpu().numpy()
    rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :].detach().cpu().numpy() 
    lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=1).mean() * 100
    rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=1).mean() * 100
    hand_jpe = (lhand_jpe+rhand_jpe)/2.0 

    return lhand_jpe, rhand_jpe, hand_jpe 


def compute_collision(ori_verts_pred,obj_name, obj_scale, obj_rot_mat, obj_trans, actual_len): 
    # ori_verts_pred: T X Nv X 3 
    # human_faces: Nf X 3 
    # obj_verts: T X Nv' X 3 
    # obj_name: string
    # obj_scale: T 
    # obj_rot_mat: T X 3 X 3 
    # obj_trans: T X 3 
    # actual_len: scalar value 

    object_sdf_folder = "/storage/group/4dvlab/congpsh/HHOI/omomo/data/rest_object_sdf_256_npy_files"

    # Load sdf 
    sdf_path = os.path.join(object_sdf_folder, obj_name+"_cleaned_simplified.obj.npy")
    sdf_data = np.load(sdf_path) # 256 X 256 X 256 

    # Convert human vertices to align with the initial object geometry. 
    tmp_verts = (ori_verts_pred - obj_trans[:, None, :]) * (1/obj_scale[:, None, None]) # T X Nv X 3 
    transformed_human_verts = torch.matmul(obj_rot_mat.transpose(1, 2), tmp_verts.transpose(1, 2)) # T X 3 X Nv     
    transformed_human_verts = transformed_human_verts.transpose(1, 2)[:actual_len] # T X Nv X 3 

    # For debug. 
    # obj_tmp_verts = (obj_verts - obj_trans[:, None, :]) * (1/obj_scale[:, None, None]) # T X Nv X 3 
    # obj_transformed_verts = torch.matmul(obj_rot_mat.transpose(1, 2), obj_tmp_verts.transpose(1, 2)) # T X 3 X Nv     
    # obj_transformed_verts = obj_transformed_verts.transpose(1, 2)[:actual_len] # T X Nv X 3 

    nv = transformed_human_verts.shape[1]

    # Load sdf json data used for querying sdf. 
    sdf_json_path = os.path.join(object_sdf_folder, obj_name+"_cleaned_simplified.obj.json")
    sdf_json_data = json.load(open(sdf_json_path, 'r'))

    if "coord_center" in sdf_json_data:
        # SIREN processed sdf
        coord_center = np.asarray(sdf_json_data['coord_center']) # 3 
        coord_min = sdf_json_data['coord_min']
        coord_max = sdf_json_data['coord_max'] 

        query_human_verts = transformed_human_verts - torch.from_numpy(coord_center)[None, None, :] # T X Nv X 3 
        query_human_verts = (query_human_verts - coord_min) / (coord_max - coord_min)
        query_human_verts -= 0.5
        query_human_verts *= 2.
    else:
        # Previous python code processed sdf 
        sdf_centroid = torch.from_numpy(np.asarray(sdf_json_data['centroid']))[None, None, :] # 1 X 1 X 3 
        sdf_extents = np.asarray(sdf_json_data['extents']) # 3 

        query_human_verts = (transformed_human_verts - sdf_centroid) * 2 / sdf_extents.max() # T X Nv X 3 
    
    query_human_verts = query_human_verts[:,:,[2, 1, 0]] # T X Nv X 3 

    vis_debug = False
            
    sdf = torch.from_numpy(sdf_data).float() # 256 X 256 X 256 

    pen_thresh = 0.04
    pen_loss = torch.tensor(0.0)

    pen_cnt = 0 

    num_steps = transformed_human_verts.shape[0]
    for t_idx in range(num_steps):    
        signed_dists = F.grid_sample(sdf.unsqueeze(0).unsqueeze(0), \
            query_human_verts[t_idx].reshape(1, nv, 1, 1, 3).float(), padding_mode='border', align_corners=True) # 
        signed_dists = signed_dists.squeeze()

        # Apply scale to the signed distance. 
        signed_dists = signed_dists * obj_scale[t_idx] 

        neg_dists_mask = signed_dists.lt(0).flatten()
        neg_dists = torch.abs(signed_dists[neg_dists_mask])
        if len(neg_dists) != 0:
            pen_mask = neg_dists.gt(pen_thresh).flatten()

            actual_neg_dists = neg_dists[pen_mask]

            if len(actual_neg_dists) > 0:
                pen_loss += actual_neg_dists.mean()
                # pen_loss += neg_dists.sum()

                pen_cnt += 1

    # import pdb 
    # pdb.set_trace() 

    if pen_cnt > 0:
        pen_loss = pen_loss/pen_cnt 
    else:
        pen_loss = torch.tensor(0.)

    pen_percent = pen_cnt/num_steps 

    # print("Pen percentage:{0}".format(pen_percent))
    # print("Pen loss:{0}".format(pen_loss.item()))
    return pen_percent, pen_loss.item() 

def compute_metrics(ori_jpos_gt, ori_jpos_pred, obj_verts):
    # verts_gt: T X Nv X 3 
    # jpos_gt: T X J X 3obj_faces
    # human_faces: Nf X 3, array  
    # obj_verts: T X No X 3
    # obj_faces: Nf X 3, array  
    # actual_len: scale value 

    # Calculate global hand joint position error 
    lhand_idx = 20
    rhand_idx = 21
    lhand_jpos_pred = ori_jpos_pred[:, lhand_idx, :]
    rhand_jpos_pred = ori_jpos_pred[:, rhand_idx, :]
    lhand_jpos_gt = ori_jpos_gt[:, lhand_idx, :]
    rhand_jpos_gt = ori_jpos_gt[:, rhand_idx, :]
    lhand_jpe = np.linalg.norm(lhand_jpos_pred - lhand_jpos_gt, axis=1).mean() * 100  #meter>>cm
    rhand_jpe = np.linalg.norm(rhand_jpos_pred - rhand_jpos_gt, axis=1).mean() * 100
    hand_jpe = (lhand_jpe+rhand_jpe)/2.0 

    # Calculate MPVPE 
    # verts_pred = ori_verts_pred - ori_jpos_pred[:, 0:1]
    # verts_gt = ori_verts_gt - ori_jpos_gt[:, 0:1]
    # verts_pred = verts_pred.detach().cpu().numpy()
    # verts_gt = verts_gt.detach().cpu().numpy()
    # mpvpe = np.linalg.norm(verts_pred - verts_gt, axis=2).mean() * 100

    # Calculate MPJPE 
    jpos_pred = ori_jpos_pred - ori_jpos_pred[:, 0:1] # zero out root
    jpos_gt = ori_jpos_gt - ori_jpos_gt[:, 0:1] 
    jpos_pred = jpos_pred
    jpos_gt = jpos_gt
    mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 100

    pred = ori_jpos_pred.clone()
    gt = ori_jpos_gt.clone()
    floor_height = determine_floor_height_and_contacts(pred.detach().cpu().numpy())
    gt_floor_height = determine_floor_height_and_contacts(gt.detach().cpu().numpy())
    
    foot_sliding_jnts = compute_foot_sliding_for_smpl(pred.detach().cpu().numpy(), floor_height)
    gt_foot_sliding_jnts = compute_foot_sliding_for_smpl(gt.detach().cpu().numpy(), gt_floor_height)

    # Compute contact score !
    num_obj_verts = obj_verts.shape[1]
    contact_threh = 0.15  

    gt_lhand_jnt = ori_jpos_gt[:, lhand_idx, :] # T X 3 
    gt_rhand_jnt = ori_jpos_gt[:, rhand_idx, :] # T X 3 


    gt_lhand2obj_dist = torch.sqrt((((gt_lhand_jnt)[:, None, :].repeat(1, num_obj_verts, 1) - (obj_verts).to((gt_lhand_jnt).device))**2).sum(dim=-1)) # T X N  
    gt_rhand2obj_dist = torch.sqrt((((gt_rhand_jnt)[:, None, :].repeat(1, num_obj_verts, 1) - (obj_verts).to((gt_rhand_jnt).device))**2).sum(dim=-1)) # T X N  

    gt_lhand2obj_dist_min = gt_lhand2obj_dist.min(dim=1)[0] # T 每一帧都记录手和物体的最近距离
    gt_rhand2obj_dist_min = gt_rhand2obj_dist.min(dim=1)[0] # T 

    gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threh)
    gt_rhand_contact = (gt_rhand2obj_dist_min < contact_threh)

    lhand_jnt = ori_jpos_pred[:, lhand_idx, :] # T X 3 
    rhand_jnt = ori_jpos_pred[:, rhand_idx, :] # T X 3 

    # print(lhand_jnt.shape)
    # print(rhand_jnt.shape)

    lhand2obj_dist = torch.sqrt((((lhand_jnt)[:, None, :].repeat(1, num_obj_verts, 1) - (obj_verts).to((lhand_jnt).device))**2).sum(dim=-1)) # T X N  
    rhand2obj_dist = torch.sqrt((((rhand_jnt)[:, None, :].repeat(1, num_obj_verts, 1) - (obj_verts).to((rhand_jnt).device))**2).sum(dim=-1)) # T X N  
   
    lhand2obj_dist_min = lhand2obj_dist.min(dim=1)[0] # T 
    rhand2obj_dist_min = rhand2obj_dist.min(dim=1)[0] # T 


    lhand_contact = (lhand2obj_dist_min < contact_threh)
    rhand_contact = (rhand2obj_dist_min < contact_threh)

    num_steps = gt_lhand_contact.shape[0]

    # Compute the distance between hand joint and object for frames that are in contact with object in GT. 
    contact_dist = 0
    gt_contact_dist = 0 

    gt_contact_cnt = 0
    contact_cnt = 0
    for idx in range(num_steps):
        if gt_lhand_contact[idx] or gt_rhand_contact[idx]:
            gt_contact_cnt += 1 
            gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_rhand2obj_dist_min[idx])
        if lhand_contact[idx] or rhand_contact[idx]:
            contact_cnt += 1 
            contact_dist += min(lhand2obj_dist_min[idx], rhand2obj_dist_min[idx]) #2只手之间更近的那个距离


    if gt_contact_cnt == 0:
        gt_contact_dist = 0 
    if contact_cnt==0:
        contact_dist = 0 

    if contact_dist != 0:
        contact_dist = contact_dist.detach().cpu().numpy()/float(contact_cnt)
    if gt_contact_dist != 0:
        gt_contact_dist = gt_contact_dist.detach().cpu().numpy()/float(gt_contact_cnt)

    #contact_percent
    gt_contact_percent = gt_contact_cnt / num_steps
    contact_percent = contact_cnt / num_steps

    # Compute precision and recall for contact. 
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for idx in range(num_steps):
        gt_in_contact = (gt_lhand_contact[idx] or gt_rhand_contact[idx]) 
        pred_in_contact = (lhand_contact[idx] or rhand_contact[idx])
        if gt_in_contact and pred_in_contact:
            TP += 1

        if (not gt_in_contact) and pred_in_contact:
            FP += 1

        if (not gt_in_contact) and (not pred_in_contact):
            TN += 1

        if gt_in_contact and (not pred_in_contact):
            FN += 1

    contact_acc = (TP+TN)/(TP+FP+TN+FN)

    if (TP+FP) == 0: # Prediction no contact!!!
        contact_precision = 0
        print("Contact precision, TP + FP == 0!!")
    else:
        contact_precision = TP/(TP+FP)
    
    if (TP+FN) == 0: # GT no contact! 
        contact_recall = 0
        print("Contact recall, TP + FN == 0!!")
    else:
        contact_recall = TP/(TP+FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0 
    else:
        contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall) 
   
    return lhand_jpe, rhand_jpe, hand_jpe, mpjpe, gt_contact_dist, contact_dist, \
    contact_precision, contact_recall, contact_acc, contact_f1_score,foot_sliding_jnts,gt_foot_sliding_jnts, \
    contact_percent,gt_contact_percent
