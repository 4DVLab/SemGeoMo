import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from manip.model.modules import CrossAttentionLayer, SelfAttentionBlock

class PointSceneMLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, widening_factor: int=1, bias: bool=True) -> None:
        super().__init__()

        self.mlp_pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, widening_factor * in_dim, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * in_dim, out_dim, bias=bias),
        )

        out_dim = out_dim * 2
        self.mlp_post = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim, bias=bias),
            nn.GELU(),
            nn.Linear(out_dim, out_dim // 2, bias=bias),
        )

    def forward(self, point_feat: torch.Tensor) -> torch.Tensor:
        point_feat = self.mlp_pre(point_feat)
        scene_feat = point_feat.mean(dim=1, keepdim=True).repeat(1, point_feat.shape[1], 1)
        point_feat = torch.cat([point_feat, scene_feat], dim=-1)
        point_feat = self.mlp_post(point_feat)

        return point_feat

class ContactMLP(nn.Module):

    def __init__(self, arch_cfg: DictConfig, contact_dim: int, point_feat_dim: int, text_feat_dim: int, time_emb_dim: int) -> None:
        super().__init__()

        self.point_mlp_dims = arch_cfg.point_mlp_dims
        self.point_mlp_widening_factor = arch_cfg.point_mlp_widening_factor
        self.point_mlp_bias = arch_cfg.point_mlp_bias

        layers = []
        idim = contact_dim + point_feat_dim + text_feat_dim + time_emb_dim
        for odim in self.point_mlp_dims:
            layers.append(PointSceneMLP(idim, odim, widening_factor=self.point_mlp_widening_factor, bias=self.point_mlp_bias))
            idim = odim
        self.point_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        if point_feat is not None:
            bs, num_points, point_feat_dim = point_feat.shape
            x = torch.cat([
                x,
                point_feat,
                language_feat.repeat(1, num_points, 1),
                time_embedding.repeat(1, num_points, 1)
            ], dim=-1) # [bs, num_points, contact_dim + point_feat_dim + language_feat_dim + time_embedding_dim]
        else:
            x = torch.cat([
                x,
                language_feat.repeat(1, num_points, 1),
                time_embedding.repeat(1, num_points, 1)
            ], dim=-1) # [bs, num_points, contact_dim + language_feat_dim + time_embedding_dim]
        x = self.point_mlp(x) # [bs, num_points, point_mlp_dim[-1]]

        return x

# arch_perceiver:
#   last_dim: 256
#   point_pos_emb: true
#   encoder_q_input_channels: 512
#   encoder_kv_input_channels: 256
#   encoder_num_heads: 8
#   encoder_widening_factor: 1
#   encoder_dropout: 0.1
#   encoder_residual_dropout: 0.0
#   encoder_self_attn_num_layers: 2

#   decoder_q_input_channels: 256
#   decoder_kv_input_channels: 512 # must be equal to encoder_q_input_channels
#   decoder_num_heads: 8
#   decoder_widening_factor: 1
#   decoder_dropout: 0.1
#   decoder_residual_dropout: 0.0

class ContactPerceiver(nn.Module):
    
    def __init__(self, contact_dim=2, point_feat_dim=256, text_feat_dim=256, time_emb_dim=100) -> None:
        super().__init__()

        self.point_pos_emb = True

        self.encoder_q_input_channels = 512
        self.encoder_kv_input_channels = 256
        self.encoder_num_heads = 8
        self.encoder_widening_factor = 1
        self.encoder_dropout = 0.1
        self.encoder_residual_dropout = 0.0
        self.encoder_self_attn_num_layers = 2
        
        self.decoder_q_input_channels = 256
        self.decoder_kv_input_channels = 512 # must be equal to encoder_q_input_channels
        self.decoder_num_heads = 8
        self.decoder_widening_factor = 1
        self.decoder_dropout = 0.1
        self.decoder_residual_dropout = 0.0

        self.language_adapter = nn.Linear(
            text_feat_dim,
            self.encoder_q_input_channels,
            bias=True)
        self.time_embedding_adapter = nn.Linear(
            time_emb_dim,
            self.encoder_q_input_channels,
            bias=True)

        # self.encoder_adapter = nn.Linear(
        #     contact_dim + point_feat_dim + (3 if self.point_pos_emb else 0), 
        #     self.encoder_kv_input_channels,
        #     bias=True)
        self.encoder_adapter = nn.Linear(
            point_feat_dim + (3 if self.point_pos_emb else 0), 
            self.encoder_kv_input_channels,
            bias=True)
        self.decoder_adapter = nn.Linear(
            self.encoder_kv_input_channels,
            self.decoder_q_input_channels,
            bias=True)

        self.encoder_cross_attn = CrossAttentionLayer(
            num_heads=self.encoder_num_heads,
            num_q_input_channels=self.encoder_q_input_channels,
            num_kv_input_channels=self.encoder_kv_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=self.encoder_self_attn_num_layers,
            num_heads=self.encoder_num_heads,
            num_channels=self.encoder_q_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.decoder_cross_attn = CrossAttentionLayer(
            num_heads=self.decoder_num_heads,
            num_q_input_channels=self.decoder_q_input_channels,
            num_kv_input_channels=self.decoder_kv_input_channels,
            widening_factor=self.decoder_widening_factor,
            dropout=self.decoder_dropout,
            residual_dropout=self.decoder_residual_dropout,
        )
       


    def forward(self, x: torch.Tensor, point_feat: torch.Tensor, language_feat: torch.Tensor, time_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, contact_dim]
            point_feat: [bs, num_points, point_feat_dim]
            language_feat: [bs, 1, language_feat_dim]
            time_embedding: [bs, 1, time_embedding_dim]
        
        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """
        if point_feat is not None:
            x = torch.cat([x, point_feat], dim=-1) # [bs, num_points, contact_dim + point_feat_dim]
        if self.point_pos_emb:
            point_pos = kwargs['c_pc_xyz']
            x = torch.cat([x, point_pos], dim=-1) # [bs, num_points, contact_dim + point_feat_dim + 3]

        # encoder
        enc_kv = self.encoder_adapter(x) # [bs, num_points, enc_kv_dim]

        language_feat = self.language_adapter(language_feat) # [bs, 1, enc_q_dim]
        time_embedding = self.time_embedding_adapter(time_embedding) # [bs, 1, enc_q_dim]
        enc_q = torch.cat([language_feat, time_embedding], dim=1) # [bs, 1 + 1, enc_q_dim]

        enc_q = self.encoder_cross_attn(enc_q, enc_kv).last_hidden_state
        enc_q = self.encoder_self_attn(enc_q).last_hidden_state

        # decoder
        dec_kv = enc_q
        dec_q = self.decoder_adapter(enc_kv) # [bs, num_points, dec_q_dim]
        dec_q = self.decoder_cross_attn(dec_q, dec_kv).last_hidden_state # [bs, num_points, dec_q_dim]

        return dec_q
