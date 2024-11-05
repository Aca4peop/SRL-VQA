import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
import torchvision.models
import torch.nn.functional as F
# import vit_pytorch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from models.vit import Transformer


class VB_CNN_Transformer(nn.Module):
    def __init__(self, img_size=14,
                 patch_size=1,
                 in_chans=1024,
                 num_classes=1,
                 embed_dim=256,
                 depths=[2, 6],
                 num_heads=[4, 16],
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 ape=False,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 attn_drop_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_econding=True,
                 con3d=False,
                 block_frame=8,
                 out_indices=3,
                 head_dim=1024):

        super().__init__()
        self.con3d = con3d
        self.conv3d = nn.Conv3d(1, 3, kernel_size=(
            block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
        self.embed_dim = embed_dim

        self.backbone = timm.create_model(
            'resnet50', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('vgg16', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('densenet161', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('resnext50_32x4d', pretrained=True, features_only=True, out_indices=[out_indices])

        # self.act = self.backbone.act1 if 'act1' in self.backbone else nn.ReLU(inplace=True)
        self.act = nn.ReLU(inplace=True)

        self.pos_econding = pos_econding
        if self.pos_econding:
            self.PosNet = PosNet(in_plane=64 + 256 + 512 +
                                 1024, out_plane=embed_dim, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # TODO: 3D-pool
        # self.pool = nn.AdaptiveAvgPool3d(1)

        self.transformer = Transformer(dim=256,
                                       depth=4,
                                       heads=8,
                                       dim_head=128,
                                       mlp_dim=head_dim,
                                       dropout=drop_rate)

        self.linear = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.Dropout(drop_rate)
        )

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if self.con3d:
            x = self.conv3d(x)
        # print(f"after con3d : {x.shape}")
        B, C, D, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        # B D C H W
        x = x.permute(0, 2, 1, 3, 4)
        # B*D C H W
        x = x.contiguous().view(B * D, C, H, W)

        # TODO : CNN-model
        x = self.backbone(x)[0]

        # TODO: 3D-pool
        # _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
        # x = x.contiguous().view(B, D, _C, _H, _W).permute(0, 2, 1, 3, 4)
        # x = self.pool(x).view(B, _C)
        # x = self.linear(x)

        # TODO: Transformer
        x = self.pool(x)
        _C, _H, _W = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.contiguous().view(B, D, _C)
        x = self.linear(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.head(x)
        return x
