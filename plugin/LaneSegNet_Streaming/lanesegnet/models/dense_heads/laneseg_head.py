#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import Linear, build_activation_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmcv.runner import auto_fp16
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.dense_heads import AnchorFreeHead

from ...utils.memory_buffer import StreamTensorMemory
from ...utils.query_update import MotionMLP
import torch.nn as nn 
import torch.nn.functional as F 
@HEADS.register_module()
class LaneSegHead(AnchorFreeHead, nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 streaming_cfg=dict(),
                 roi_size=(102.4,51.2),
                 num_query=200,
                 with_box_refine=False,
                 with_shared_param=None,
                 transformer=None,
                 bbox_coder=None,
                 num_reg_fcs=2,
                 code_weights=None,
                 num_points=10,
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,
                 pts_dim=3,
                 sync_cls_avg_factor=False,
                 num_lane_type_classes=3,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_mask=dict(type='CrossEntropyLoss', loss_weight=3.0),
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_lane_type=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
                     )),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 pred_mask=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called, also need nn.Module.
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_lane_type = build_loss(loss_lane_type)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pred_mask = pred_mask
        self.loss_mask_type = loss_mask['type']
        self.num_points = num_points

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        if loss_lane_type.use_sigmoid:
            self.cls_lane_type_out_channels = num_lane_type_classes
        else:
            self.cls_lane_type_out_channels = num_lane_type_classes + 1

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        self.with_box_refine = with_box_refine
        if with_shared_param is not None:
            self.with_shared_param = with_shared_param
        else:
            self.with_shared_param = not self.with_box_refine
        self.as_two_stage = False

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = pts_dim * 30
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.code_size
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.gt_c_save = self.code_size

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_reg_fcs = num_reg_fcs
        self.num_lane_type_classes = num_lane_type_classes
        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']
        else: 
            self.streaming_query = False
        
        if self.streaming_query:
            self.batch_size = streaming_cfg['batch_size']
            self.topk_query = streaming_cfg['topk']
            self.trans_loss_weight = streaming_cfg.get('trans_loss_weight', 0.0)
            self.query_memory = StreamTensorMemory(self.batch_size,)
            self.reference_points_memory = StreamTensorMemory(self.batch_size,)
            c_dim = 12 
            self.query_update = MotionMLP(c_dim=c_dim,f_dim=self.embed_dims, identity=True)
            self.target_memory = StreamTensorMemory(self.batch_size)
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2,-roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32)) 


        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size // 3))
        reg_branch = nn.Sequential(*reg_branch)

        reg_branch_offset = []
        for _ in range(self.num_reg_fcs):
            reg_branch_offset.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch_offset.append(nn.ReLU())
        reg_branch_offset.append(Linear(self.embed_dims, self.code_size // 3))
        reg_branch_offset = nn.Sequential(*reg_branch_offset)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

        cls_left_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_left_type_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_left_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_left_type_branch.append(nn.ReLU(inplace=True))
        cls_left_type_branch.append(Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_left_type = nn.Sequential(*cls_left_type_branch)        

        cls_right_type_branch = []
        for _ in range(self.num_reg_fcs):
            cls_right_type_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_right_type_branch.append(nn.LayerNorm(self.embed_dims))
            cls_right_type_branch.append(nn.ReLU(inplace=True))
        cls_right_type_branch.append(Linear(self.embed_dims, self.cls_lane_type_out_channels))
        fc_cls_right_type = nn.Sequential(*cls_right_type_branch)

        mask_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))

        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if not self.with_shared_param:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.reg_branches_offset = _get_clones(reg_branch_offset, num_pred)
            self.cls_left_type_branches = _get_clones(fc_cls_left_type, num_pred)
            self.cls_right_type_branches = _get_clones(fc_cls_right_type, num_pred)
            self.mask_embed = _get_clones(mask_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.reg_branches_offset = nn.ModuleList(
                [reg_branch_offset for _ in range(num_pred)])
            self.cls_left_type_branches = nn.ModuleList(
                [fc_cls_left_type for _ in range(num_pred)])
            self.cls_right_type_branches = nn.ModuleList(
                [fc_cls_right_type for _ in range(num_pred)])
            self.mask_embed = nn.ModuleList(
                [mask_branch for _ in range(num_pred)])

    def _forward_mask_head(self, output, bev_feats, lvl):
        # shape (bs, num_query, embed_dims)
        bev_feats = bev_feats.view([bev_feats.shape[0], self.bev_h, self.bev_w, self.embed_dims])
        bev_feats = bev_feats.permute(0, 3, 1, 2).contiguous()
        mask_embed = self.mask_embed[lvl](output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, bev_feats)
        return outputs_mask

    def propagate(self, query_embedding, img_metas, return_loss=True):
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = [] # stil empty so cannot stack 

        tmp = self.query_memory.get(img_metas) 
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas'] 

        tmp =self.reference_points_memory.get(img_metas) 
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        # print('ref_pts_memory',len(ref_pts_memory)) # list
        # print('pose_memory',len(pose_memory)) # list


        if return_loss: 
            target_memory = self.target_memory.get(img_metas)['tensor']
            trans_loss = query_embedding.new_zeros((1, )) # transformation loss 
            num_pos = 0 # for what 
        # first frame check 
        is_first_frame_list = tmp['is_first_frame']

        for i in range(bs): 
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                padding = query_embedding.new_zeros((self.topk_query, self.embed_dims))
                propagated_query_list.append(padding)

                padding = query_embedding.new_zeros((self.topk_query, self.num_points, 2))
                prop_reference_points_list.append(padding)
                # print("->1st frame: propagated_query_list", padding.shape)
            else: 
                # use float64 to do precise coord transformation
                # translation and rotation on roi (60,30)
                #in openlanev2: Agoverse ego2global -> lidar2global
                # print('huhu')
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['can_bus'][:3], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['lidar2global_rotation'], dtype=torch.float64) # translation vector
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['can_bus'][:3], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['lidar2global_rotation'], dtype=torch.float64) # translation vector
                
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix # transformation   
                pos_encoding = prev2curr_matrix.float()[:3].view(-1) 

                prop_q = query_memory[i]
                # print('->prop_q', prop_q.shape)
                # MotionMLP
                query_memory_updated = self.query_update(
                    prop_q,
                    pos_encoding.view(1,-1).repeat(len(query_memory[i]), 1)
                )   
                # print("->query_memory_updated", query_memory_updated.shape)           
                propagated_query_list.append(query_memory_updated.clone())

                pred = self.reg_branches[-1](query_memory_updated).sigmoid()
                #TODO: num_points: num_points in polyline for map element bucket
                # how to adapt for lane segment prediction?
                # lanesegment: unidirectional - 10 points polyline
                assert list(pred.shape) == [self.topk_query, 2*self.num_points]
                
                if return_loss: 
                    targets = target_memory[i]
                    weights = targets.new_ones((self.topk_query, 2*self.num_points))
                    bg_idx = torch.all(targets.view(self.topk_query,-1) == 0.0, dim=1)
                    num_pos = num_pos + (self.topk_query - bg_idx.sum())
                    weights[bg_idx,:] = 0.0 

                    denormed_targets = targets * self.roi_size + self.origin # (topk, numpts, 2)
                    denormed_targets = torch.cat([
                        denormed_targets,
                        denormed_targets.new_zeros((self.topk_query,self.num_points,1)),
                        denormed_targets.new_ones((self.topk_query,self.num_classes,1))
                    ], dim=-1) 
                    assert list(denormed_targets.shape) == [self.topk_query, self.num_points, 4]
                    curr_targets = torch.einsum('lk,ijk->ijl',prev2curr_matrix.float(),denormed_targets)
                    normed_targets = (curr_targets[...,:2] - self.origin) / self.roi_size # (num_prop, num_points, 2)
                    normed_targets = torch.clip(normed_targets,min=0.,max=1.).reshape(-1,2*self.num_points)
                    trans_loss += self.loss_reg(pred, normed_targets,weights, avg_factor=1.0)
                # ref pts 
                prev_ref_pts = ref_pts_memory[i]
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
                assert list(prev_ref_pts.shape) == [self.topk_query,self.num_points, 2]
                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((self.topk_query,self.num_points,1)),#z-axis
                    denormed_ref_pts.new_ones((self.topk_query,self.num_points,1))# 4th dim
                ],dim=-1) # (num_prop, num_pts, 4)
                assert list(denormed_ref_pts.shape) == [self.topk_query, self.num_points, 4] 

                curr_ref_pts = torch.einsum('lk,ijk->ijl',prev2curr_matrix,denormed_ref_pts.double()).float()
                normed_ref_pts = (curr_ref_pts[...,:2] - self.origin) / self.roi_size # (num_prop, num_pts, 2) 
                normed_ref_pts = torch.clip(normed_ref_pts, min=0.,max=1.)
                # print('->normed_ref_pts', normed_ref_pts.shape)
                prop_reference_points_list.append(normed_ref_pts)
        # print(len(propagated_query_list))
        # print(len(prop_reference_points_list))
        # print('huhu')
        prop_query_embedding = torch.stack(propagated_query_list) # (bs, topk, embed_dims) 
        prop_ref_pts = torch.stack(prop_reference_points_list) # (bs, topk, num_pts, 2) 
        assert list(prop_query_embedding.shape) == [bs, self.topk_query, self.embed_dims]
        assert list(prop_ref_pts.shape) == [bs, self.topk_query, self.num_points, 2] 

        # init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts) # init ref -> sigmoid 
        # init_reference_points = init_reference_points.view(bs, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
        memory_query_embedding = None

        if return_loss:
            trans_loss = self.trans_loss_weight * trans_loss / (num_pos + 1e-10)
            return query_embedding, prop_query_embedding, prop_ref_pts, memory_query_embedding, is_first_frame_list, trans_loss
        else:
            # test mode: no loss calculation 
            return query_embedding, prop_query_embedding, prop_ref_pts, memory_query_embedding, is_first_frame_list      

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, bev_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_lanes_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 99].
            all_mask_preds (Tensor): Sigmoid outputs from the segmentation \
                head with normalized value in the range of [0,1].
                Shape []

        streammapnet

                inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            prop_query=prop_query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            prop_reference_points=prop_ref_pts,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            is_first_frame_list=is_first_frame_list,
            query_key_padding_mask=query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn,
        )

        """
        dtype = mlvl_feats[0].dtype
        B, N, C, H, W = mlvl_feats[0].shape 
        img_masks = bev_feats.new_zeros((B,H,W)) 
        pos_embed = None 
        query_embedding = self.query_embedding.weight[None, ...].repeat(B,1,1) # B, num_Q, embed_dims 
        input_query_num = self.num_query 
        if self.streaming_query: 
            query_embedding, prop_query_embedding, prop_ref_pts, memory_query, is_first_frame_list, trans_loss = \
                self.propagate(query_embedding, img_metas,return_loss=True) 
        else: 
            # go to transformer for ident init 
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(B)]



        object_query_embeds = self.query_embedding.weight.to(dtype)
        # print("before laneseg transformer, object_query_embeds",object_query_embeds.shape)
        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            # mlvl_pos_embeds=,            
            # mlvl_feats=[bev_feats,],
            object_query_embed=object_query_embeds,
            is_first_frame_list=is_first_frame_list, 
            prop_reference_points=prop_ref_pts,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=(self.reg_branches, self.reg_branches_offset) if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches,
            img_metas=img_metas,
            memory_query=None,
            prop_query=prop_query_embedding, 
        )
        
        hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        if not self.training: # forward test 
            reference = inter_references[-1] # reference from final decoder layer 
            reference = inverse_sigmoid(reference) # activation 
            assert reference.shape[-1] == self.pts_dim # pts_dim 

            outputs_class = self.cls_branches[-1](hs[-1]) # 
            output_left_type = self.cls_left_type_branches[-1](hs[-1])
            output_right_type = self.cls_right_type_branches[-1](hs[-1])

            tmp = self.reg_branches[-1](hs[-1]) # output of regression branch 
            bs, num_query, _ = tmp.shape # tmp.shape 
            tmp = tmp.view(bs, num_query, -1, self.pts_dim) # tmp.view bs, 200, -1, pts_dims 
            tmp = tmp + reference # reg + reference 
            tmp = tmp.sigmoid() # inverse sigmoid then sigmoid ???
 
            coord = tmp.clone() 
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            centerline = coord.view(bs, num_query, -1).contiguous()

            offset = self.reg_branches_offset[-1](hs[-1])
            left_laneline = centerline + offset
            right_laneline = centerline - offset

            # segmentation head
            if self.pred_mask:
                outputs_mask = self._forward_mask_head(hs[-1], bev_feats, -1)

            outputs_classes = torch.stack([outputs_class])
            outputs_coords = torch.stack([torch.cat([centerline, left_laneline, right_laneline], axis=-1)])
            output_left_types = torch.stack([output_left_type])
            output_right_types = torch.stack([output_right_type])

            outs = {
                'all_cls_scores': outputs_classes,
                'all_lanes_preds': outputs_coords,
                'all_mask_preds': torch.stack([outputs_mask]) if self.pred_mask else None,
                'all_lanes_left_type': output_left_types,
                'all_lanes_right_type': output_right_types,
                'history_states': hs
            }

            return outs

        outputs_classes = []
        outputs_coord = []
        outputs_masks = []
        output_left_types = []
        output_right_types = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            assert reference.shape[-1] == self.pts_dim

            outputs_class = self.cls_branches[lvl](hs[lvl])
            output_left_type = self.cls_left_type_branches[lvl](hs[lvl])
            output_right_type = self.cls_right_type_branches[lvl](hs[lvl])

            tmp = self.reg_branches[lvl](hs[lvl])
            bs, num_query, _ = tmp.shape
            tmp = tmp.view(bs, num_query, -1, self.pts_dim)
            tmp = tmp + reference
            tmp = tmp.sigmoid()

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            centerline = coord.view(bs, num_query, -1).contiguous()

            offset = self.reg_branches_offset[lvl](hs[lvl]) # offset from mlvl, hs???? 
            left_laneline = centerline + offset   # left laneline
            right_laneline = centerline - offset 

            # segmentation head
            outputs_mask = self._forward_mask_head(hs[lvl], bev_feats, lvl) 

            outputs_classes.append(outputs_class)
            outputs_coord.append(torch.cat([centerline, left_laneline, right_laneline], axis=-1))
            outputs_masks.append(outputs_mask)
            output_left_types.append(output_left_type)
            output_right_types.append(output_right_type)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coord)
        outputs_masks = torch.stack(outputs_masks)
        output_left_types = torch.stack(output_left_types)
        output_right_types = torch.stack(output_right_types)        

        outs = {
            'all_cls_scores': outputs_classes,
            'all_lanes_preds': outputs_coords,
            'all_mask_preds': outputs_masks,
            'all_lanes_left_type': output_left_types,
            'all_lanes_right_type': output_right_types,
            'history_states': hs
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           lanes_pred,
                           masks_pred,
                           lanes_left_type_preds,
                           lanes_right_type_preds,
                           gt_labels,
                           gt_lanes,
                           gt_instance_masks,
                           gt_lanes_left_type, 
                           gt_lanes_right_type,
                           gt_bboxes_ignore=None):

        num_bboxes = lanes_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(lanes_pred, masks_pred, cls_score, gt_lanes, 
                                             gt_instance_masks, gt_labels)

        sampling_result = self.sampler.sample(assign_result, lanes_pred, gt_lanes)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lanes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_lanes.new_ones(num_bboxes)

        labels_left_type = gt_lanes_left_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_left_type[pos_inds] = gt_lanes_left_type[sampling_result.pos_assigned_gt_inds].long()
                
        labels_right_type = gt_lanes_right_type.new_full((num_bboxes,), self.num_lane_type_classes, dtype=torch.long)
        labels_right_type[pos_inds] = gt_lanes_right_type[sampling_result.pos_assigned_gt_inds].long()
        
        # bbox targets
        gt_c = gt_lanes.shape[-1]
        if gt_c == 0:
            gt_c = self.gt_c_save
            sampling_result.pos_gt_bboxes = torch.zeros((0, gt_c)).to(sampling_result.pos_gt_bboxes.device)
        else:
            self.gt_c_save = gt_c

        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds] = 1.0

        # mask targets
        mask_targets = gt_instance_masks[pos_assigned_gt_inds]
        mask_weights = masks_pred.new_zeros((self.num_query, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_weights, labels_left_type, labels_right_type,
                pos_inds, neg_inds, pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    masks_preds_list,
                    lanes_left_type_preds_list,
                    lanes_right_type_preds_list,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type_list, 
                    gt_lanes_right_type_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, lanes_targets_list,
         lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
         pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, lanes_preds_list, masks_preds_list, lanes_left_type_preds_list, lanes_right_type_preds_list,
            gt_labels_list, gt_lanes_list, gt_masks_list, gt_lanes_left_type_list, gt_lanes_right_type_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )

        return (labels_list, label_weights_list, lanes_targets_list,
                lanes_weights_list, masks_targets_list, masks_weights_list, labels_left_type_list, labels_right_type_list,
                num_total_pos, num_total_neg, assign_result)
    
    def loss_single(self,
                    cls_scores,
                    lanes_preds,
                    masks_preds,
                    lanes_left_type, 
                    lanes_right_type,
                    gt_lanes_list,
                    gt_labels_list,
                    gt_masks_list,
                    gt_lanes_left_type, 
                    gt_lanes_right_type,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]
        mask_preds_list = [masks_preds[i] for i in range(num_imgs)]
        lanes_left_type_list = [lanes_left_type[i] for i in range(num_imgs)]
        lanes_right_type_list = [lanes_right_type[i] for i in range(num_imgs)]
        cls_reg_seg_targets = self.get_targets(cls_scores_list, lanes_preds_list, mask_preds_list, lanes_left_type_list, lanes_right_type_list,
                                           gt_lanes_list, gt_labels_list, gt_masks_list, gt_lanes_left_type, gt_lanes_right_type,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, mask_weights_list, labels_left_type_list, labels_right_type_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_seg_targets

        labels = torch.cat(labels_list, 0)
        labels_left_type = torch.cat(labels_left_type_list, 0)
        labels_right_type = torch.cat(labels_right_type_list, 0)        
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        mask_targets = torch.cat(mask_targets_list, 0)
        mask_weights = torch.stack(mask_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        lanes_left_type = lanes_left_type.reshape(-1, self.cls_lane_type_out_channels)
        lanes_right_type = lanes_right_type.reshape(-1, self.cls_lane_type_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        loss_left_type = self.loss_lane_type(
            lanes_left_type, labels_left_type, label_weights, avg_factor=cls_avg_factor)
        
        loss_right_type = self.loss_lane_type(
            lanes_right_type, labels_right_type, label_weights, avg_factor=cls_avg_factor)
        loss_lane_type = loss_left_type + loss_right_type
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))

        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan, :self.code_size], 
            bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        # segmentation part
        cls_scores = cls_scores.flatten(0,1)
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones, shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = masks_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            loss_dice = masks_preds.sum()
            loss_mask = masks_preds.sum()
            return loss_cls, loss_bbox, loss_dice, loss_mask, assign_result

        # dice loss
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks
        )

        # mask loss (point based, deprecated)
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = mask_targets.reshape(-1)
        if self.loss_mask_type == 'FocalLoss':
            mask_targets = (1 - mask_targets).long()
        if self.loss_mask_type == 'CrossEntropyLoss':
            mask_targets = mask_targets.reshape(mask_preds.shape).bool()

        loss_mask = self.loss_mask(
            mask_preds, mask_targets, avg_factor=num_total_masks * h * w
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_dice = torch.nan_to_num(loss_dice)
            loss_mask = torch.nan_to_num(loss_mask)

        return loss_cls, loss_bbox, loss_dice, loss_mask, loss_lane_type, assign_result

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             gt_lanes_3d,
             gt_labels_list,
             gt_instance_masks,
             gt_lane_left_type,
             gt_lane_right_type,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_instance_masks (list[Tensor]): Ground truth instance masks for each lane segment 
                of map size with shape (num_gts, 100, 50)
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_lanes_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                all_masks_preds (Tensor): Bitwise instance segmentation outputs of 
                    all decoder layers. Each is a bitwise segmentation map with shape (100,50).
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_lanes_preds = preds_dicts['all_lanes_preds']
        all_mask_preds = preds_dicts['all_mask_preds']
        all_lanes_left_type = preds_dicts['all_lanes_left_type']
        all_lanes_right_type = preds_dicts['all_lanes_right_type']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_lanes_list = [lane for lane in gt_lanes_3d]

        all_gt_lanes_list = [gt_lanes_list for _ in range(num_dec_layers)]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        gt_masks_list = [mask for mask in gt_instance_masks]

        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]

        all_gt_lanes_left_type = [gt_lane_left_type for _ in range(num_dec_layers)]

        all_gt_lanes_right_type = [gt_lane_right_type for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_dice, losses_mask, losses_lane_type, assign_result = multi_apply(
            self.loss_single, all_cls_scores, all_lanes_preds, all_mask_preds, all_lanes_left_type, all_lanes_right_type,
            all_gt_lanes_list, all_gt_labels_list, all_gt_masks_list, all_gt_lanes_left_type, all_gt_lanes_right_type)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]
        loss_dict['loss_seg_dice'] = losses_dice[-1]
        loss_dict['loss_seg_mask'] = losses_mask[-1]
        loss_dict['loss_lane_type'] = losses_lane_type[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_dice_i, loss_mask_i, loss_lane_type_i in zip(losses_cls[:-1],
                                                                                       losses_bbox[:-1],
                                                                                       losses_dice[:-1],
                                                                                       losses_mask[:-1],
                                                                                       losses_lane_type[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_seg_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_seg_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_lane_type'] = loss_lane_type_i
            num_dec_layer += 1

        return loss_dict, assign_result

    @force_fp32(apply_to=('preds_dicts'))
    def get_lanes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bbox of lane prediction, lane detection, not segmentation 
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            lanes = preds['lane3d']
            scores = preds['scores']
            labels = preds['labels']
            result = [lanes, scores, labels]
            if 'left_type_scores' in preds:
                left_type_scores = preds['left_type_scores']
                left_type_labels = preds['left_type_labels']
                right_type_scores = preds['right_type_scores']
                right_type_labels = preds['right_type_labels']
                result.extend([left_type_scores, left_type_labels, right_type_scores, right_type_labels])
            ret_list.append(result)

        return ret_list
