#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import torch

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head, build_neck, build_backbone
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from plugin.LaneSegNet_FastBEV_New.lanesegnet.utils.grid_mask  import GridMask

from ...utils.builder import build_bev_constructor


@DETECTORS.register_module()
class LaneSegNet(MVXTwoStageDetector):

    def __init__(self,
                 img_view_transformer, # FastRayTransformer 
                 img_bev_encoder_backbone=None, # CustomResNet
                 img_bev_encoder_neck=None, # FPN_LSS
                 use_grid_mask=False,
                 use_depth=False,
                 lane_head=None,
                 lclc_head=None,
                 bbox_head=None,
                 lcte_head=None,
                 video_test_mode=False, # modified
                 **kwargs):

        super(LaneSegNet, self).__init__(**kwargs)

        if img_view_transformer is not None:
            self.bev_constructor = build_neck(img_view_transformer)

        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = build_neck(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = build_neck(img_bev_encoder_neck)
        self.use_depth = use_depth

        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head)
        else:
            self.pts_bbox_head = None
        
        if lclc_head is not None:
            self.lclc_head = build_head(lclc_head)
        else:
            self.lclc_head = None

        if bbox_head is not None:
            bbox_head.update(train_cfg=self.train_cfg.bbox)
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None

        if lcte_head is not None:
            self.lcte_head = build_head(lcte_head)
        else:
            self.lcte_head = None

        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs) # Resnet 
        stereo_feat = None
        if stereo:
            # No stereo 
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat
    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x) # CustomResNet 
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x
    def prepare_inputs(self, img, img_metas):
        # split the inputs into each frame

        # print('->len inputs:',len(inputs))
        # assert len(inputs) == 7
        exit()
        B, N, C, H, W = img.shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs
        

        img_metas['sensor2egos'] = img_metas['sensor2egos'].view(B, N, 4, 4)

        img_metas['ego2globals'] = img_metas['ego2globals'].view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        return img, img_metas

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        img = self.prepare_inputs(img, img_metas) # a list 
        x, _ = self.image_encoder(img[0]) # 
        x, depth = self.img_view_transformer([x] + img[1:8])
        x = self.bev_encoder(x)
        return [x], depth
    
    # def extract_img_feat(self, img, img_metas, len_queue=None):
        # """Extract features of images."""
        # B = img.size(0)
        # if img is not None:

        #     if img.dim() == 5 and img.size(0) == 1:
        #         img.squeeze_()
        #     elif img.dim() == 5 and img.size(0) > 1:
        #         B, N, C, H, W = img.size()
        #         img = img.reshape(B * N, C, H, W)
        #     img_feats = self.img_backbone(img)

        #     if isinstance(img_feats, dict):
        #         img_feats = list(img_feats.values())
        # else:
        #     return None
        # if self.with_img_neck:
        #     img_feats = self.img_neck(img_feats)

        # img_feats_reshaped = []
        # for img_feat in img_feats:
        #     BN, C, H, W = img_feat.size()
        #     if len_queue is not None:
        #         img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
        #     else:
        #         img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        # return img_feats_reshaped

    @auto_fp16(apply_to=('img'))

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)
    # def extract_feat(self, img, img_metas=None, len_queue=None):
    #     img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
    #     return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None, 

                      img=None,
                      img_metas=None,
                      gt_lanes_3d=None,
                      gt_lane_labels_3d=None,
                      gt_lane_adj=None,
                      gt_lane_left_type=None,
                      gt_lane_right_type=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_instance_masks=None,
                      gt_bboxes_ignore=None,
                      ):

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]


        # img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img, img_metas=img_metas)        
        
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)

        losses = dict()
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas)
        loss_inputs = [outs, gt_lanes_3d, gt_lane_labels_3d, gt_instance_masks, gt_lane_left_type, gt_lane_right_type]
        lane_losses, lane_assign_result = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        for loss in lane_losses:
            losses['lane_head.' + loss] = lane_losses[loss]
        lane_feats = outs['history_states']

        if self.lclc_head is not None:
            lclc_losses = self.lclc_head.forward_train(lane_feats, lane_assign_result, lane_feats, lane_assign_result, gt_lane_adj)
            for loss in lclc_losses:
                losses['lclc_head.' + loss] = lclc_losses[loss]

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list

    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)

        bev_feats = self.bev_constructor(x, img_metas, prev_bev)
        outs = self.pts_bbox_head(x, bev_feats, img_metas)

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        if self.lclc_head is not None:
            lane_feats = outs['history_states']
            lsls_results = self.lclc_head.get_relationship(lane_feats, lane_feats)
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]

        return bev_feats, lane_results, lsls_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, lane_results, lsls_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, lane, lsls in zip(results_list, lane_results, lsls_results):
            result_dict['lane_results'] = lane
            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = lsls
            result_dict['lste_results'] = None

        return new_prev_bev, results_list
