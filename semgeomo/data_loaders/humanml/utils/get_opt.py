import os
from argparse import Namespace
import re
from os.path import join as pjoin
from .word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    #print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
    elif opt.dataset_name == "behave":
        print("behave")
        opt.data_root ="/storage/group/4dvlab/congpsh/HHOI/behave_t2m/"#############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'pred_text')
        opt.joint_dir = pjoin(opt.data_root, 'new_joints')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 200
    elif opt.dataset_name == "omomo": #!
        print("omomo")
        opt.data_root ="/storage/group/4dvlab/congpsh/HHOI/OMOMO"#############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs_fps15')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs_fps15')
        opt.joint_dir = pjoin(opt.data_root, 'new_joints')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        #opt.text_dir = pjoin(opt.data_root, 'pred_text2')
        #opt.text_dir = pjoin(opt.data_root, 'fine_texts')
        #opt.text_dir = pjoin(opt.data_root, 'fine_texts2')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 100
    elif opt.dataset_name == "Hodome": #!
        print("dome")
        opt.data_root ="/storage/group/4dvlab/congpsh/HHOI/Dome/mocap_ground/"#############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')  #fps15/30
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 100
    elif opt.dataset_name == "imhoi":
        opt.data_root ="/storage/group/4dvlab/IM-HOI/dataset/" #############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts_200')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 100
    elif opt.dataset_name == "interx":
        opt.data_root ="/storage/group/4dvlab/congpsh/HHOI/Inter-X/Inter-X_Dataset/" #############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 200
    elif opt.dataset_name == "intergen":
        opt.data_root ="/storage/group/4dvlab/congpsh/HHOI/InterGen/" #############################
        opt.npy_root=pjoin(opt.data_root, 'new_joint_vecs')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 50
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt