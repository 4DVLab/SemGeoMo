import numpy as np
import json
import torch
HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',#10
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',#20
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

HML_TRAJ_MASK = np.zeros_like(HML_ROOT_MASK)
HML_TRAJ_MASK[1:3] = True

NUM_HML_FEATS = 263

def get_control_mask(mask_name, shape, **kwargs):
    assert mask_name == "global_joint", "mask_name must be 'global_joint', got {}".format(mask_name)
    mask = np.zeros(shape)
    mask = np.maximum(mask, get_global_joint_mask(shape, **kwargs))
    return mask


def select_random_indices(bs, seq_len, num_selected):
    indices = []
    for _ in range(bs):
        indices.append(np.random.choice(seq_len, size=num_selected, replace=False))
    return np.array(indices)


def get_global_joint_mask(shape, joint, ratio=1, dataset = 'humanml'):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_joint (22 for HumanML3D), 3, seq_len))
    """
    bs, num_joint, joint_dim, seq_len = shape
    assert joint_dim == 3, "joint_dim must be 3, got {}".format(joint_dim)
    if dataset in [ 'humanml',"behave","omomo","imhoi","interx","intergen","Unify","Hodome"]:
        assert num_joint == 22, "num_joint must be 22, got {}".format(num_joint)
        if joint == 'all':
            random_joint = np.random.randint(0, num_joint,  size=(1,bs))
        elif joint == 'random_two':
            random_joint = np.random.randint(0, num_joint,  size=(2,bs))
        elif joint == 'random_three':
            random_joint = np.random.randint(0, num_joint,  size=(3,bs))
        else:
            if type(joint) != list:
                assert joint in HML_JOINT_NAMES, "joint must be one of {}, got {}".format(HML_JOINT_NAMES, joint)
                random_joint = (np.ones((1,bs), dtype=int) * HML_JOINT_NAMES.index(joint))
            else:
                random_joint_1 = (np.ones((1,bs), dtype=int) * HML_JOINT_NAMES.index(joint[0]))
                random_joint_2 = (np.ones((1,bs), dtype=int) * HML_JOINT_NAMES.index(joint[1]))
                random_joint = np.concatenate((random_joint_1, random_joint_2), axis=0)
    elif dataset == 'kit':
        assert num_joint == 21, "num_joint must be 21, got {}".format(num_joint)
        if joint == 'all':
            random_joint = np.random.randint(0, num_joint, size=(bs,))
        elif joint == 'pelvis':
            random_joint = np.zeros((bs,), dtype=int)
        else:
            raise NotImplementedError("joint must be one of {} in kit dataset, got {}".format(['all', 'pelvis'], joint))
    else:
        raise NotImplementedError("dataset must be one of {}, got {}".format(['humanml', 'kit'], dataset))
    if np.abs(1 - ratio) < 1e-3:
        random_t = np.ones((bs, 1, 1, seq_len))
    else:
        num_selected = int(ratio * seq_len)
        random_t = np.zeros((bs, 1, 1, seq_len))
        selected_indices = select_random_indices(bs, seq_len, num_selected)
        random_t[np.arange(bs)[:, np.newaxis], :, :, selected_indices] = 1

    random_t = np.tile(random_t, (1, 1, 3, 1))
    mask = np.zeros(shape)
    for i in range(random_joint.shape[0]):
        mask[np.arange(bs)[:, np.newaxis], random_joint[i, :, np.newaxis], :, :] = random_t.astype(float)
    return mask

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
    


    
