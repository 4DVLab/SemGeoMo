import torch
import torch.nn as nn
import copy
from .mdm import MDM
from .pointtransformer import PointTransformerFeatureExtractor
from .pointnet_util import PointNetSetAbstraction
import sys
sys.path.append("..")
from data_loaders.humanml.scripts.motion_process import recover_from_ric


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers, norm=None)

    def forward_with_condition(self, src, list_of_controlnet_output, mask=None, src_key_padding_mask=None):
        output = src
        for mod, control_feat in zip(self.layers, list_of_controlnet_output):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = output + control_feat
        if self.norm is not None:
            output = self.norm(output)
        return output

    def return_all_layers(self, src, mask=None, src_key_padding_mask=None):
        output = src
        all_layers = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            all_layers.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return all_layers

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, ff_dim=1024, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (bs, T, feature_dim)
        x = x.permute(1, 0, 2)  # (T, bs, feature_dim) for transformer input
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (bs, T, feature_dim)
        return x
        
class CrossAttention(nn.Module):  
    def __init__(self, d_model, n_heads):  
        super(CrossAttention, self).__init__()  
        self.query_linear = nn.Linear(d_model, d_model)  
        self.key_linear = nn.Linear(d_model, d_model)  
        self.value_linear = nn.Linear(d_model, d_model)  
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads)  
        self.output_linear = nn.Linear(d_model, d_model)  

    def forward(self, query, key, value):  
        # Project input features into query, key, value  
        query_proj = self.query_linear(query)  # [seqlen, bs, d]  
        key_proj = self.key_linear(key)         # [seqlen, bs, d]  
        value_proj = self.value_linear(value)   # [seqlen, bs, d]  
        
        # Use multihead attention  
        attn_output, _ = self.attention(query_proj, key_proj, value_proj)  
        
        # Project output back to the original dimension  
        output = self.output_linear(attn_output)  
        return output 
        
class ControlMDM(MDM):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):

        super(ControlMDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)
        self.args = args
        self.num_layers = num_layers
        self.multi_person = args.multi_person
        self.upper_orientation_index = [0, 16, 17]  # root, l_shoulder, r_shoulder
        self.lower_orientation_index = [0, 1, 2]  # root, l_hip, r_hip

        #3 mlp+ projection
        self.mlp1 = nn.Sequential(
            nn.Linear(2, 64),  
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512)
        )
        #pc_bps
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            )
        self.bps_encoder = self.bps_encoder.to("cuda:0")
        self.pc_mlp = nn.Linear(in_features=3+256, out_features=512)

        self.temporal_transformer = TemporalTransformer(input_dim=512, num_heads=4, ff_dim=1024, num_layers=2).to("cuda")
        self.cross_attention_layer = CrossAttention(d_model=512, n_heads=4).to("cuda")
        self.further_fusion = False  ###
        if self.further_fusion:
            self.final_ca_layer = CrossAttention(d_model=512, n_heads=4).to("cuda")
        # linear layers init with zeros
        if self.dataset == 'kit':
            self.first_zero_linear = nn.Linear(21*3*2 + 2*3, self.latent_dim)
        elif self.dataset in [ 'humanml',"behave","omomo","imhoi","interx","intergen","Unify"]:
            self.first_zero_linear = nn.Linear(22*3, self.latent_dim) ####
        else:
            raise NotImplementedError('Supporting only kit and humanml dataset, got {}'.format(self.dataset))
        
        nn.init.zeros_(self.first_zero_linear.weight)
        nn.init.zeros_(self.first_zero_linear.bias)
        self.mid_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)])
        for m in self.mid_zero_linear:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        if self.arch == 'trans_enc':
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            del self.seqTransEncoder
            self.seqTransEncoder_mdm = TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
            self.seqTransEncoder_control = TransformerEncoder(seqTransEncoderLayer,
                                                             num_layers=self.num_layers)
        else:
            raise ValueError('Supporting only trans_enc arch.')

        self.freeze_block(self.input_process)
        self.freeze_block(self.sequence_pos_encoder)
        self.freeze_block(self.seqTransEncoder_mdm)
        self.freeze_block(self.embed_timestep)
        if 'text' in self.cond_mode:
            self.freeze_block(self.embed_text)
        self.freeze_block(self.output_process)

    def inv_transform(self, data):
        assert self.std is not None and self.mean is not None
        #assert data.requires_grad == True
        std = torch.tensor(self.std, dtype=data.dtype, device=data.device, requires_grad=False)
        mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device, requires_grad=False)
        output = torch.add(torch.mul(data, std), mean)
        return output
    
    def compute_triangle_normals(self, triangles):
        # Compute the vectors from the first point to the other two points
        v1 = triangles[:,:, 1] - triangles[:, :,0]
        v2 = triangles[:,:, 2] - triangles[:,:,0]

        # Compute the cross product of v1 and v2 to get the normal vectors
        normals = torch.cross(v2, v1, dim=-1)

        # Normalize the normal vectors to unit length
        normals = nn.functional.normalize(normals, dim=-1)
        return normals
    
    def humanml_to_global_joint(self, x):
        n_joints = 22 if x.shape[1] == 263 else 21
        curr_joint = self.inv_transform(x.permute(0, 2, 3, 1)).float()
        assert curr_joint.shape[1] == 1
        curr_joint = recover_from_ric(curr_joint, n_joints)
        curr_joint = curr_joint.view(-1, *curr_joint.shape[2:]).permute(0, 2, 3, 1)
        return curr_joint
        
    def affordance_pc(self,dist,bps,pc):
        f = []
        self.mlp1 = self.mlp1.to("cuda:0")
        self.mlp2 = self.mlp2.to("cuda:0")
        self.mlp3 = self.mlp3.to("cuda:0")
        self.max_pool = self.max_pool.to("cuda:0")
        self.projection = self.projection.to("cuda:0")
        self.pc_mlp = self.pc_mlp.to("cuda:0")
        
        bps = torch.stack([torch.tensor(arr).float() for arr in bps] ) #BSXTX1024X3
        dist = torch.stack([torch.tensor(arr).float() for arr in dist] )  #BSXTX1024X2
        com_pos = torch.stack([torch.tensor(arr.mean(1)).float() for arr in pc] ).to("cuda:0") #BSXTX3
        
        for index in range(0,len(bps)):  #len bs
            x = dist[index].float()    #T X 1024 X 2
            T, N, _ = x.shape
            x = x.view(T * N, 2)  # T*1024 X 2
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = x.view(T, N, 256)  #  T X 1024 X 256
            x = x.permute(0, 2, 1)  # T X 256 X 1024
            x = self.max_pool(x).squeeze(-1)  # T X 256
            x = self.projection(x)  # T X 512
            f.append(x)
        dist_p=torch.stack([arr for arr in f])  #bs x T x 512
        
        bps = bps.reshape(bps.shape[0],bps.shape[1],-1) #bs x T x (1024*3)
        enc_bps = self.bps_encoder(bps)  #bs x T x256
        pc_cond = torch.cat((com_pos, enc_bps), dim=-1) # BS X T X (3+256) 
        pc_p = self.pc_mlp(pc_cond)  #bs x T x 512
        
        affordance = dist_p + pc_p # bs x T x 512
        
        
        return affordance   #bs X T X 512

    def forward(self, x, timesteps, y=None):
        
        #print("forward-----")
        
        bs, njoints, nfeats, seqlen = x.shape
        control_bs, n_global_joints, xyz_dim, control_frames = y['global_joint'].shape
        assert bs == control_bs and seqlen == control_frames, "bs {} != {} or seqlen {} != {}".format(bs, control_bs, seqlen, control_frames)
        assert xyz_dim ==3, "xyz_dim {} != 3".format(xyz_dim)
        # prepare global joints for controlmdm
        curr_joint = self.humanml_to_global_joint(x).clone().detach()  # [bs, njoints, 3, seqlen]
        curr_joint.requires_grad = False

        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        
        #print(y.keys())
        
        force_mask = y.get('uncond', False)

        ca = False
        fine_coarse = True
        long_clip = True
        concate = True  
        if long_clip:
            time_emb = copy.deepcopy(emb)
            emb_fine = self.encode_text_longclip(y['fine_text']).unsqueeze(0).float()
            emb += self.embed_text(self.mask_cond(emb_fine, force_mask=force_mask))
            if fine_coarse:
                enc_text_coarse = self.encode_text(y['text']).unsqueeze(0)
                enc_text_coarse = self.mask_cond(enc_text_coarse, force_mask=force_mask)
                if concate:
                    enc_text_coarse += time_emb
                    emb = torch.cat((enc_text_coarse, emb), axis=0)
                else:
                    emb += enc_text_coarse
            
        elif fine_coarse:
            enc_text = self.encode_text_pe(y['fine_text'],device=emb.device).permute(1,0,2)
            emb_fine = emb.repeat(enc_text.shape[0],1,1)
            emb_fine +=  self.embed_text(enc_text) # 77, bsx512
            
            enc_text_coarse = self.encode_text(y['text']).unsqueeze(0) # 1xbsx512
            # print(enc_text_coarse.shape,enc_text_coarse.shape)
            emb += self.embed_text(self.mask_cond(enc_text_coarse, force_mask=force_mask))
            
            emb = torch.cat((emb, emb_fine), axis=0)  # [77 +1, bs, d]
            
        elif 'text' in self.cond_mode and ca:
            enc_text = self.encode_text_pe(y['fine_text'],device=emb.device).permute(1,0,2)
            # print("enc_text========",enc_text.shape) #1xbsx512
            # print(emb.shape)
            emb = emb.repeat(enc_text.shape[0],1,1)
            emb += self.embed_text(enc_text) # 77, bsx512
            # print("emb========",emb.shape) #1xbsx512
        
        
        elif 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
       
        dist = y['dist']
        bps = y['bps']
        pc = y['pc']
        affordance_map = self.affordance_pc(dist,bps,pc)
        affordance_map = self.temporal_transformer(affordance_map)
        affordance_map = affordance_map.permute(1,0,2) # T x  bs x 512
        mask = []
        for i in range(affordance_map.shape[0]):
            mask.append(self.mask_cond(affordance_map[i],force_mask=force_mask))
        feature =  torch.stack([arr.float() for arr in mask]).to(emb.device) # [seqlen, bs, d]

        # relative position to control-joint(two-hand)
        relative_position = ((y['global_joint'].float() - curr_joint)*y['global_joint_mask'].bool().float()).float()  # [bs, njoints, 3, seqlen]
        relative_position = relative_position.permute(3, 0, 1, 2).reshape(control_frames, control_bs, -1)  # [seqlen, bs, 22*3]]
        global_joint_feat = relative_position # [seqlen, bs, 22*3]
        control = self.first_zero_linear(global_joint_feat).to(emb.device)  # [seqlen, bs, d]
        
        # fusion
        fused_features = self.cross_attention_layer(query=feature, key=control, value=control)
        fused_features = fused_features.to(emb.device)

        
        if self.further_fusion:
            attn_output = self.final_ca_layer(emb,fused_features,fused_features)
            enc_text_coarse = self.encode_text(y['text']).unsqueeze(0)
            enc_text_coarse = self.mask_cond(enc_text_coarse, force_mask=force_mask)
            emb = torch.cat((emb, enc_text_coarse,attn_output,fused_features), axis=0)
        else:
            emb = torch.cat((emb, fused_features), axis=0)  # [seqlen +1, bs, d]
        

        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]
        emb = emb.to(x.device)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+seqlen+1, bs, d]
        
        control_output_list = self.seqTransEncoder_control.return_all_layers(xseq)  
        for i in range(self.num_layers):
            control_output_list[i] = self.mid_zero_linear[i](control_output_list[i])
        
        output = self.seqTransEncoder_mdm.forward_with_condition(xseq, control_output_list)[-100:]
        output = self.output_process(output)  # [bs, njoints, nfeats, seqlen]
        #print(output.shape)
        return output

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]
        # return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    
    def trainable_parameter_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False

    def unfreeze_block(self, block):
        block.train()
        for p in block.parameters():
            p.requires_grad = True
    
    def forward_without_control(self, x, timesteps, y=None):   #
        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        # Embed motion to latent space (frame by frame)
        x = self.input_process(x) #[seqlen, bs, d]
        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder_mdm(xseq)[1:]  # [seqlen, bs, d]
        output = self.output_process(output)  # [bs, njoints, nfeats, seqlen]
        return output