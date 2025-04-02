#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.runner.base_module import BaseModule

from ...utils.builder import BEV_CONSTRUCTOR
from ....bevformer.modules.temporal_self_attention import TemporalSelfAttention
from ....bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from ....bevformer.modules.decoder import CustomMSDeformableAttention


@BEV_CONSTRUCTOR.register_module()
class BEVFormerConstructer(BaseModule):
    """Implements the BEVFormer BEV Constructer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 bev_h=200,
                 bev_w=200,
                 rotate_center=[100, 100],
                 encoder=None,
                 positional_encoding=None,
                 **kwargs):
        super(BEVFormerConstructer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.rotate_center = rotate_center

        self.init_layers()

    def init_layers(self):
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
 
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # @auto_fp16(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, **kwargs):
        """
        obtain bev features.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # print('->bev_former: mlvl shape', mlvl_feats[0].shape)

        bev_queries = self.bev_embedding.weight.to(dtype) # bev_embedding 
        # print('->bev_former: 1.bev_queries', bev_queries.shape)
        
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        # print('->bev_former: 2.bev_queries', bev_queries.shape)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype) # mask 
        # print('->bev_former: bev_mask', bev_mask.shape)

        bev_pos = self.positional_encoding(bev_mask).to(dtype) 
        # print('->bev_former: 1.bev_pos', bev_pos.shape)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # pos_embed 
        # print('->bev_former: 2.bev_pos', bev_pos.shape)
        # BEVFormer assumes the coords are x-right and y-forward for the nuScenes lidar
        # but OpenLane-V2's coords are x-forward and y-left
        # here is a fix for any lidar coords, the shift is calculated by the rotation matrix
        delta_global = np.array([each['can_bus'][:3] for each in img_metas])# what is can_bus 
        # exit()
        lidar2global_rotation = np.array([each['lidar2global_rotation'] for each in img_metas])
        delta_lidar = []
        for i in range(bs):
            delta_lidar.append(np.linalg.inv(lidar2global_rotation[i]) @ delta_global[i])
        delta_lidar = np.array(delta_lidar)
        
        shift_y = delta_lidar[:, 1] / self.real_h
        shift_x = delta_lidar[:, 0] / self.real_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        shift = bev_queries.new_tensor([shift_x,shift_y])
        
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy
        print('############### debug ##############')

        ##### (1,3)  (1,3)  (1,3,3)
        ##### 1       1     1 
        print(shift.shape)
        print(delta_global.shape, delta_lidar.shape, lidar2global_rotation.shape)
        print(len(delta_global), len(delta_lidar), len(lidar2global_rotation))

        # print(img_metas)
        # print(prev_bev);exit()
        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = img_metas[i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in img_metas])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape # [1,7,256, 40,20]
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            print('1. feat in mlvl', feat.shape)
            if self.use_cams_embeds:
                print('cams_embeds', self.cams_embeds.shape)
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            print('1. level_embeds', self.level_embeds.shape)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        print('feat_flatten[0] shape:', feat_flatten[0].shape)
        feat_flatten = torch.cat(feat_flatten, 2)
        print('feat_flatten after cat', feat_flatten.shape)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims). (7,H,W,1,256)

        

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten, # with itself 
            feat_flatten, # ? 
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas,
            **kwargs
        )
        return bev_embed
    # bev_feats = self.bev_constructor.forward_trt(img_feats_reshaped,prev_bev,can_bus,lidar2img,img_shape, use_prev_bev)

    def forward_trt(self, mlvl_feats, can_bus, lidar2global_rotation,lidar2img, img_shape=None, use_prev_bev=None):
        """
        TODO: determine the shape of 
        - can_bus: np.ndarray 
        - lidar2global 
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        prev_bev = None

        bev_queries = self.bev_embedding.weight.to(dtype) # bev_embedding        
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype) # mask 

        bev_pos = self.positional_encoding(bev_mask).to(dtype) 
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)  # pos_embed 
        # BEVFormer assumes the coords are x-right and y-forward for the nuScenes lidar
        # but OpenLane-V2's coords are x-forward and y-left
        # here is a fix for any lidar coords, the shift is calculated by the rotation matrix

        delta_global = torch.tensor(can_bus[:, :3], dtype=torch.float32, requires_grad=False)
        delta_lidar = []
        for i in range(bs):
            inv_rot = torch.inverse(lidar2global_rotation[i])
            # delta = torch.matmul(inv_rot, delta_global[i])
            delta = inv_rot @ delta_global[i]
            delta_lidar.append(delta)
        delta_lidar = torch.stack(delta_lidar, dim=0)
        ##### (1,3)  (1,3)  (1,3,3)
        # print(delta_global.shape, delta_lidar.shape, lidar2global_rotation.shape)


        
        # numpy version 

        shift_y = delta_lidar[:, 1] / self.real_h
        shift_x = delta_lidar[:, 0] / self.real_w
        shift_y = shift_y * int(self.use_shift)
        shift_x = shift_x * int(self.use_shift)

        # shift = bev_queries.new_tensor([shift_x, shift_y])
        shift = torch.stack([shift_x, shift_y], dim=0).permute(1,0)
        # print('##',shift.shape);exit()
        # shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = can_bus[i][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # can_bus = can_bus.unsqueeze(0)
        # print('can_bus:',can_bus.shape);exit()
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * int(self.use_can_bus)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape # [1,7,256, 40,20]
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            print('1. feat in mlvl', feat.shape)
            if self.use_cams_embeds:
                print('cams_embeds', self.cams_embeds.shape)
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            print('1. level_embeds', self.level_embeds.shape)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        print('feat_flatten[0] shape:', feat_flatten[0].shape)
        feat_flatten = torch.cat(feat_flatten, 2)
        print('feat_flatten after cat', feat_flatten.shape)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims). (7,H,W,1,256)

        

        bev_embed = self.encoder.forward_trt(
            bev_queries,
            feat_flatten, # with itself 
            feat_flatten, # ? 
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            lidar2img=lidar2img,
            img_shape=img_shape,
        )
        return bev_embed
