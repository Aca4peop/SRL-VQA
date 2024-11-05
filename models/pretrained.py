import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint
import torchvision.models
import torch.nn.functional as F
# import vit_pytorch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.vit import Transformer


class pre_trained_model(nn.Module):
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
                        pos_econding=False,
                        con3d=False,
                        block_frame=8,
                        out_indices=3,
                        head_dim=1024,
                        block_num=4):

        super().__init__()
        self.con3d = con3d
        self.conv3d = nn.Conv3d(1, 3, kernel_size=(block_frame, 3, 3), stride=(block_frame, 1, 1), padding=(0, 1, 1))
        self.embed_dim = embed_dim

        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('densenet161', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('vgg16', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('resnext50_32x4d', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('densenet121', pretrained=True, features_only=True, out_indices=[4])
        # self.backbone = timm.create_model('wide_resnet50_2', pretrained=True, features_only=True, out_indices=[out_indices])
        # self.backbone = timm.create_model('resnest50d_1s4x24d', pretrained=True, features_only=True, out_indices=[3])
        # self.backbone = timm.create_model('wide_resnet101_2', pretrained=True, features_only=True, out_indices=[out_indices])

        # self.act = self.backbone.act1 if 'act1' in self.backbone else nn.ReLU(inplace=True)
        self.act = nn.ReLU(inplace=True)

        # self.pos_econding = pos_econding
        # if self.pos_econding:
        #     self.PosNet = PosNet(in_plane=64 + 256 + 512 + 1024, out_plane=embed_dim, kernel_size=3)

        # TODO: 3D-pool
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.pool = nn.AdaptiveAvgPool3d(1)

        self.transformer = Transformer(dim=256, 
                                    depth=4, 
                                    heads=8, 
                                    dim_head=128, 
                                    mlp_dim=head_dim, 
                                    dropout=drop_rate)
        
        # self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.Dropout(drop_rate)
        )

        self.bit_rate = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 1)
        )

        self.dis_type = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 7)
            # nn.Linear(256, 5)
        )

        self.fps = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 3)
        )


    def forward(self, ref, dis):
        if self.con3d:
            ref = self.conv3d(ref)
            dis = self.conv3d(dis)
            # x = self.act(x)
        # print(f"after con3d : {x.shape}")
        # B, C, D, H, W = ref.shape[0], ref.shape[1], ref.shape[2], ref.shape[3], ref.shape[4]
        B, C, D, H, W = ref.size()
        # B D C H W
        ref = ref.permute(0, 2, 1, 3, 4)
        dis = dis.permute(0, 2, 1, 3, 4)
        # B*D C H W
        ref = ref.contiguous().view(B * D, C, H, W) 
        dis = dis.contiguous().view(B * D, C, H, W) 

        # TODO : CNN-model
        ref = self.backbone(ref)[0]
        dis = self.backbone(dis)[0]

        # TODO: 3D-pool
        # _C, _H, _W = ref.shape[-3], ref.shape[-2], ref.shape[-1]
        # ref = ref.contiguous().view(B, D, _C, _H, _W).permute(0, 2, 1, 3, 4)
        # dis = dis.contiguous().view(B, D, _C, _H, _W).permute(0, 2, 1, 3, 4)
        # dis = self.pool(dis).view(B, _C)
        # ref = self.pool(ref).view(B, _C)
        
        # ref = self.linear(ref)
        # dis = self.linear(dis)        

        # TODO: Transformer
        dis = self.pool(dis)
        ref = self.pool(ref)
        
        _C, _H, _W = ref.shape[-3], ref.shape[-2], ref.shape[-1]
        ref = ref.contiguous().view(B, D, _C)
        dis = dis.contiguous().view(B, D, _C)
        ref = self.linear(ref)
        dis = self.linear(dis)
        # print(f"batch : {B}, depth : {D}, channel : {}")

        # TODO : transformer
        ref = self.transformer(ref)
        dis = self.transformer(dis)
        ref = ref.mean(dim=1)
        dis = dis.mean(dim=1)
        # TODO: END

        bit_rate = self.bit_rate(ref - dis)
        dis_type = self.dis_type(ref - dis)
        fps = self.fps(dis)

        # return  bit_rate, fps 
        return bit_rate, dis_type, fps


