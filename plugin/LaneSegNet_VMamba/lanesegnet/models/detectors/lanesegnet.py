#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head, build_backbone
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from ...utils.builder import build_bev_constructor
# from ....backbones import build_vssm_model, build_model

@DETECTORS.register_module()
class LaneSegNet(MVXTwoStageDetector):

    def __init__(self,
                 bev_constructor=None,
                 lane_head=None,
                 lclc_head=None,
                 bbox_head=None,
                 lcte_head=None,
                 img_backbone = None, 
                 video_test_mode=False, # modified
                 **kwargs):
        # super init here is for what 
        super(LaneSegNet, self).__init__(**kwargs)

        if img_backbone is not None: 
            # VMambaT
            self.img_backbone = build_backbone(img_backbone)
            
        # this is the bev encoder 
        if bev_constructor is not None:
            self.bev_constructor = build_bev_constructor(bev_constructor)
        # LaneSegHead 
        if lane_head is not None:
            lane_head.update(train_cfg=self.train_cfg.lane)
            self.pts_bbox_head = build_head(lane_head) # points bbox head 
        else:
            self.pts_bbox_head = None
        # relationship head 
        if lclc_head is not None:
            self.lclc_head = build_head(lclc_head)
        else:
            self.lclc_head = None
        # 
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

        # should incorporate query update into this
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        if self.video_test_mode: 
            # init video module here 


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images.
        img.shape = [0,7,3,800,1024]
        0: 
        7: number of camera views
        3: Number of channels 
        800: height 
        1024: width 
        Mamba: [48,3,3,3]
        Resnet: batchsize, num images, channels, height,width
        Vmamb: 
        
        """
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                # B = 0 
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                
                B, N, C, H, W = img.size()
                # B * 7, C, H, W 
                img = img.reshape(B * N, C, H, W) # concat 7 images 
                # print('img after reshape', img.shape)
            # elif img.size(0) == 0: 
            #     # when B = 0, there is no data 
            #     exit
            
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            # feed through FPN, len(img_feats = 3)
            # [7,192,40,52]: VMambaT
            # ([7, 512, 40, 52]): Resnet50
            # print('->img_feats 0 ', img_feats[0].shape)
            # print('->img_feats 1 ', img_feats[1].shape)
            # print('->img_feats 2 ', img_feats[2].shape)
            # Add depthwise convolution to reduce channels to 128, code later 


            
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        #TODO: Try to use temporal information, previous frames are effective for occluded object detection, 
        may be effective for map detection 
        """
        self.eval()
        # do not track gradient when backprop
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.bev_constructor(img_feats, img_metas, prev_bev)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
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

        len_queue = img.size(1) # B,C,H,W -> C
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) # how effective is this?
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)      # Backbone
        bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev) # BEVFormerConstructor 

        losses = dict()
        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas) # LaneSegHead takes in img_feats and img_feats
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
            # without video test mode 
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
        """Test function
        x: input
        img_metas: test images metainfo
        img: what is it then 
        """
        batchsize = len(img_metas)

        bev_feats = self.bev_constructor(x, img_metas, prev_bev) # BEVFormerConstructor
        outs = self.pts_bbox_head(x, bev_feats, img_metas)       # LaneSegHead decoder output
        # forward_train: pts_bbox_head(img_feats, bev_feats, img_metas)                    

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale) # get coordinate of polyline 

        if self.lclc_head is not None:
            lane_feats = outs['history_states'] # history state? 
            lsls_results = self.lclc_head.get_relationship(lane_feats, lane_feats) # Relationship Head get topo of lanes 
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]

        return bev_feats, lane_results, lsls_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentation.
        forward_test -> simple_test -> simple_test_pts
        img used for feature extractor, not necessary after that  
        
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas) # backbone 

        results_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, lane_results, lsls_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        for result_dict, lane, lsls in zip(results_list, lane_results, lsls_results):
            result_dict['lane_results'] = lane
            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = lsls
            result_dict['lste_results'] = None

        return new_prev_bev, results_list
