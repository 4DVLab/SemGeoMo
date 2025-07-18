import torch
import sys
sys.path.append("..")
from diffusion.control_diffusion import ControlGaussianDiffusion
from model.cfg_sampler import wrap_model
from model.mdm import MDM
from model.ControlMDM import ControlMDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_controlmdm_and_diffusion(args, data, device, ModelClass, DiffusionClass=ControlGaussianDiffusion):  
    model, diffusion = create_model_and_diffusion(args, ModelClass=ModelClass, DiffusionClass=DiffusionClass)
    model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.mean = data.dataset.t2m_dataset.mean
    model.std = data.dataset.t2m_dataset.std

    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args)
    return model, diffusion

def load_model(args, data, device, ModelClass=MDM):
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=ModelClass)
    model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args)
    return model, diffusion

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    #print(missing_keys)
    if 't_pos_encoder.pe' in missing_keys:
        missing_keys.remove('t_pos_encoder.pe')
    if 't_pos_encoder.pe' in unexpected_keys:
        unexpected_keys.remove('t_pos_encoder.pe')
   # assert len(unexpected_keys) == 0
    #assert all([k.startswith('clip_model.') for k in missing_keys])

def load_pretrained_mdm(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


'''def load_pretrained_mdm_to_controlmdm(model, state_dict): #for finetue
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    transformer_encoder_weight = {}
    #for key, value in state_dict.items():
        #if key.startswith('seqTransEncoder'):
            #transformer_encoder_weight[key[16:]] = value
            #unexpected_keys.remove(key)
    model.seqTransEncoder_mdm.load_state_dict(transformer_encoder_weight, strict=False)
    model.seqTransEncoder_control.load_state_dict(transformer_encoder_weight, strict=False)
    #assert len(unexpected_keys) == 0
    #assert all([k.startswith('clip_model.') for k in missing_keys])
    #print("The following parameters are trained from scratch.")
    #for k in missing_keys:
        #if not k.startswith('clip_model.') and not k.startswith('seqTransEncoder'):
            #print(k)'''
            
            
def load_pretrained_mdm_to_controlmdm(model, state_dict): #for pretrain-mdm
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    transformer_encoder_weight = {}
    for key, value in state_dict.items():
         if key.startswith('seqTransEncoder'):
             transformer_encoder_weight[key[16:]] = value
             unexpected_keys.remove(key)
    model.seqTransEncoder_mdm.load_state_dict(transformer_encoder_weight, strict=False)
    model.seqTransEncoder_control.load_state_dict(transformer_encoder_weight, strict=False)
     
    #print(unexpected_keys)
    #assert len(unexpected_keys) == 0
    #assert all([k.startswith('clip_model.') for k in missing_keys])
    print("The following parameters are trained from scratch.")
    for k in missing_keys:
        if not k.startswith('clip_model.') and not k.startswith('seqTransEncoder'):
            print(k)


def load_split_mdm(model, state_dict, cutting_point):
    new_state_dict = {}
    orig_trans_prefix = 'seqTransEncoder.'
    for k, v in state_dict.items():
        if k.startswith(orig_trans_prefix):
            orig_layer = int(k.split('.')[2])
            orig_suffix = '.'.join(k.split('.')[3:])
            target_split = 'seqTransEncoder_start.' if orig_layer < cutting_point else 'seqTransEncoder_end.'
            target_layer = orig_layer if orig_layer < cutting_point else orig_layer - cutting_point
            new_k = target_split + 'layers.' + str(target_layer) + '.' + orig_suffix
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


def create_model_and_diffusion(args, ModelClass=MDM, DiffusionClass=SpacedDiffusion):
    model = ModelClass(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion

def get_model_args(args):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'text' if args.dataset in ['humanml', 'kit','babel', 'pw3d',"behave","imhoi","omomo","interx","intergen","Unify","Hodome"] else 'action'

    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml', 'pw3d',"behave","imhoi","omomo","interx","intergen","Unify","Hodome"]:
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'babel':
        data_rep = 'rot6d'
        njoints = 135
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    else:
        raise TypeError(f'dataset {args.dataset} is not currently supported')

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'diffusion-steps': args.diffusion_steps, 'batch_size': args.batch_size, 'use_tta': args.use_tta,
            'trans_emb': args.trans_emb, 'concat_trans_emb': args.concat_trans_emb, 'args': args}


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion): #create diffusion
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps  #steps from the args
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    
    print(f"number of diffusion-steps: {steps}")

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if hasattr(args, 'multi_train_mode'):
        multi_train_mode = args.multi_train_mode
    else:
        multi_train_mode = None

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        batch_size=args.batch_size,
        multi_train_mode=multi_train_mode,
        bfgs_times_first = args.bfgs_times_first,
        bfgs_times_last = args.bfgs_times_last,
        bfgs_interval = args.bfgs_interval,
    )