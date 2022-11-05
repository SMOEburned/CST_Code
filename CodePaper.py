# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:59:37 2022

@author: Seyd Teymoor Seydi and Mojtaba Sadegh

     
"""


#------------------------------------- Load ----------------------------------------------------------------

import torch
from torch import nn
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
print(torch.__version__)

#------------------------------------- Define Model Functions ----------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels=8, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BATCHnorm1= nn.BatchNorm2d(out_channels, affine=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BATCHnorm2= nn.BatchNorm2d(out_channels, affine=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.BATCHnorm3= nn.BatchNorm2d(out_channels, affine=False)
    def forward(self, x):
        x0 = F.relu(self.conv1(x))
        x0=self.BATCHnorm1(x0)

        x1 = F.relu(self.conv2(x0))
        x1=self.BATCHnorm2(x1) 
        x2 =x0+x1

        x3 = F.relu(self.conv3(x2))
        x3=self.BATCHnorm3(x3) 
        x4=  x2  +x3             
        return x4
 class UpScaleSubpixel(nn.Module):
    def __init__(self, SCALE=scale_Value):
        super(UpScaleSubpixel, self).__init__()
        self.SCL = nn.PixelShuffle(upscale_factor=SCALE)
    def forward(self, x):
        x = self.SCL(x)
        return x   
    
class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups=1):
        super(GroupNorm, self).__init__(num_groups, num_channels)

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class ShiftViTBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=12,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 input_resolution=None):
        super(ShiftViTBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.n_div = n_div

    def forward(self, x):
        x = self.shift_feat(x, self.n_div)
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return 

    @staticmethod
    def shift_feat(x, n_div):
        B, C, H, W = x.shape
        g = C // n_div
        out = torch.zeros_like(x)

        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  

        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  
        return out

class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, (2, 2), stride=2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return 
class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 n_div=12,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=None,
                 norm_layer=None,
                 downsample=None,
                 use_checkpoint=False,
                 act_layer=nn.GELU):

        super(BasicLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            ShiftViTBlock(dim=dim,
                          n_div=n_div,
                          mlp_ratio=mlp_ratio,
                          drop=drop,
                          drop_path=drop_path[i],
                          norm_layer=norm_layer,
                          act_layer=act_layer,
                          input_resolution=input_resolution)
            for i in range(depth)
        ])


        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return 


class PatchEmbed(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=32,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ShiftViT(nn.Module):

    def __init__(self,
                 n_div=12,
                 img_size=56,
                 patch_size=4,
                 in_chans=16,
                 num_classes=2,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer='GN1',
                 act_layer='GELU',
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        assert norm_layer in ('GN1', 'BN')
        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'GN1':
            norm_layer = partial(GroupNorm, num_groups=1)
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=False)
        else:
            raise NotImplementedError

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio


        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.conv_layer = ConvLayer()
        self.SCLE =UpScaleSubpixel()

        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]


        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               n_div=n_div,
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               act_layer=act_layer)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) \
            if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x) 
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):

        x = self.conv_layer(x)

        x = self.SCLE(x)

        
        x = self.forward_features(x)
        x = self.head(x)
        x = F.sigmoid(x)
        return x
#-------------------------------------Model Define----------------------------------------------------------------
        
model = ShiftViT(embed_dim=, depths=, mlp_ratio=, drop_path_rate=, n_div=)

