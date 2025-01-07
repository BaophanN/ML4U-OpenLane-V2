#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import copy
import numpy as np
import math 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sys import exit
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head, build_neck
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn import build_conv_layer
from mmseg.ops import resize 
# from ...utils.builder import build_bev_constructor


@DETECTORS.register_module()
class LaneSegNet(MVXTwoStageDetector):
    def __init__(self,
                 voxel_size,
                 n_voxels, 
                #  bev_constructor=None,
                 neck_3d=None,
                 lane_head=None,
                 lclc_head=None,
                 bbox_head=None,
                 lcte_head=None,
                 video_test_mode=False, # modified
                 neck_fuse=None,
                 backproject='inplace',
                 multi_scale_id=None, 
                 multi_scale_3d_scaler=None,
                 extrinsic_noise=0,
                 style='v4',
                 **kwargs):

        super(LaneSegNet, self).__init__(**kwargs)

        if neck_3d is not None:
            self.neck_3d = build_neck(neck_3d)
        # if bev_constructor is not None:
        #     self.bev_constructor = build_bev_constructor(bev_constructor)
  
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1) 
        # modify from above 
        # self.neck_fuse = build_conv_layer(neck_fuse)
        self.style = style
        self.backproject = backproject
        ### Post add    
        self.multi_scale_id = multi_scale_id 
        self.multi_scale_3d_scaler = multi_scale_3d_scaler 
        self.n_voxels = n_voxels 
        self.voxel_size = voxel_size     
        ###   
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
        self.extrinsic_noise = extrinsic_noise 
        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["cam_intrinsics"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]) # lidar2img 
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
            print('img_meta key',img_meta.keys())
            print('->intrinsic',intrinsic)
            print('->extrinsic',extrinsic)
            print('->projection',projection)
            exit()
        return torch.stack(projection)

    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        x = self.img_backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.img_neck(x) # FPN
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        # if self.with_cp and x.requires_grad:
        #     mlvl_feats = cp.checkpoint(_inner_forward, x)
        # else:
        # img_feats = self.img_neck(img_feats)

        img_feats = _inner_forward(x)
        img_feats = list(img_feats)
###
        img_feats_reshaped = [] 
        for img_feat in img_feats: 
            BN,C,H,W = img_feat.size() 
            if len_queue is not None: 
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else: 
                img_feats_reshaped.append(img_feat.view(B, int(BN/B),C,H,W))
        mlvl_feats = list(img_feats) # need this for creating features 3d
        features_2d = img_feats_reshaped # this for feeding the decoder
###
        # features_2d = None
        # if self.bbox_head_2d:
        #     features_2d = mlvl_feats

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([B, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w]): len = seq

            # seg=1: [bs, nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 7, dim=1)

            volume_list = []
            # temporal frames 
            for seq_id in range(len(mlvl_feat_split)):
                # only 1 sequence
             
                volumes = []
                # many batches, seg_img_meta in enumerate
  
                for batch_id, seq_img_meta in enumerate(img_metas):
                    """
                    Traverse all the samples in img_metas, only get img_metas of the final image
                    img_metas = [each[len_queue - 1] for each in img_metas]
                    batch_id, img_meta, each separate 
                    [sample1, sample2, sample3, sample4]: 4 samples / gpu 
                    batch_id, img_meta 
                    """
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    # print('->seq_img_meta',seq_img_meta)
                    # exit()
                    # many sequences 
                    img_meta = copy.deepcopy(seq_img_meta)
                    # print('forward train img_meta', img_meta)
                    # exit()
                    # img_meta = img_meta[0]
                    # Uncomment this
                    img_meta["lidar2img"] = img_meta["lidar2img"][seq_id*7:(seq_id+1)*7]
                    # print(img_meta)
                    # print('->type of',type(img_meta['img_shape'])) # list 
                    # exit()
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id*7:(seq_id+1)*7]
                    #     # why? 
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    # print('->height',img_meta["img_shape"]) # 775,775,3
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(np.array([0, 0, -0.6])),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes
        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out
            
        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        x = _inner_forward(x)

        return x, None, features_2d # list with 4 scales

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """
        img_feats:  resnet + neck + view transform
        valids: 0 
        mlvl_feats: resnet + neck 

        """
        bev_feats, valids, img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return bev_feats, valids, img_feats

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
        # print('->forward train, img shape', img.shape)
        # exit()
        # # # # print('->forward_train. len img_metas', len(img_metas))
        # # # # print('->forward train, img metas[0]', len(img_metas[0]))
        # # # # print(img_metas[0][1]['filename'])
        # # # # print(img_metas[0]['ori_shape'])
        #sys.exit()
        # # # # print(f'img_metas: {img_metas}')
        # B,N,C,H,W  
        prev_img = img[:, :-1, ...] # 
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
        else:
            prev_bev = None
        # comment this 
        # print('->forward train',img_metas)
        img_metas = [each[len_queue-1] for each in img_metas]
        # print('after each', img_metas)
        # exit()
        # # # print('->type img_metas', type(img_metas))
        # for batch_id, img_meta in enumerate(img_metas): 
        #     # print(batch_id)
        #     # print(img_meta.keys())
        # exit()
        bev_feats, valids, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # this is bev_feats right now 

        # bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        # # # # print(f'len img_metas, {len(img_metas)}')
        losses = dict()
        # print(f'bev_feats, {[feat.shape for feat in bev_feats]}') 
        # print(f'img_feats {[feat.shape for feat in img_feats]}')
        outs = self.pts_bbox_head(mlvl_feats=img_feats, bev_feats=bev_feats, img_metas=img_metas)
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
        # print('img shape',img.shape)
        # img_metas = [{batch_id:each} for batch_id, each in enumerate(img_metas)]
        # for batch_id, img_meta in enumerate(img_metas): 
            # print(batch_id)
            # print(img_meta.keys())
        # exit()
        # # # # print(f'forward_test->len img, {len(img)}')
        # # # # print(f'->img[0] shape:{img[0].shape}') # 7,3,800,1024
        # # # # print(f'->len img_metas:{len(img_metas)}') # 
        """
        forward_train 
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        """
        # # # print('->forward test, img shape', img.shape)
        # # # print('->forward test, img metas cak', len(img_metas[0]))
        # # # print(img_metas[0]['ori_shape'])
        # sys.exit()
        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # exit()
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
            img_metas[0]['can_bus'][-1] = 0 # This since no temporal info used 
            img_metas[0]['can_bus'][:3] = 0

        new_prev_bev, results_list = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results_list
    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentation."""
        # bev_feats, _, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        """
        extract_feat(self, img, img_metas=None, len_queue=None) returns 
        """

        """
        bev_feats, valids, img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return bev_feats, valids, img_feats
        """
        results_list = [dict() for i in range(len(img_metas))]
        # # # # print(f'->img_metas: {img_metas}')
        new_prev_bev, lane_results, lsls_results = self.simple_test_pts(
            img_metas, img, prev_bev, rescale=rescale)
        #   x,       , 1 set of 7 imgs, 
        for result_dict, lane, lsls in zip(results_list, lane_results, lsls_results):
            result_dict['lane_results'] = lane
            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = lsls
            result_dict['lste_results'] = None

        return new_prev_bev, results_list
    def simple_test_pts(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)
        # print('simple_test_pts,len img_metas', batchsize)
        # # # # print('->batchsize len', batchsize)
        # exit()
        bev_feats, valids, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # bev_feats = self.bev_constructor(x, img_metas, prev_bev)
        # outs = self.pts_bbox_head(mlvl_feats=img_feats, bev_feats=bev_feats, img_metas=img_metas)

        outs = self.pts_bbox_head(img_feats, bev_feats, img_metas)

        lane_results = self.pts_bbox_head.get_lanes(
            outs, img_metas, rescale=rescale)

        if self.lclc_head is not None:
            lane_feats = outs['history_states']
            lsls_results = self.lclc_head.get_relationship(lane_feats, lane_feats)
            lsls_results = [result.detach().cpu().numpy() for result in lsls_results]
        else:
            lsls_results = [None for _ in range(batchsize)]

        return bev_feats, lane_results, lsls_results



@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    """
    n_voxels: [200,200,4]
    voxel_size: 0.5,0.5,1.5
    origin: 
    """
    # print(f'n_voxels:{n_voxels},voxel_size:{voxel_size},origin:{origin}')
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    # print('origin',origin.shape)
    new_origin = origin - n_voxels / 2.0 * voxel_size
    # 200,200,4 x 
    # # # # print(f'get_points, points, {points.shape}')
    # # # # print(f'get_points, voxel_size, {voxel_size.shape}')
    # # # # print(f'get_points, new_origin, {new_origin.shape}')
    # print('new origin',new_origin.shape)
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1) # [0.5,0.5, 1.5] + lidar2global_translation [x,y,z]
    # print('get_points returns points',points.shape)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    # # print('->feature shape',features.shape)
    # # print('->points shape',points.shape)
    # # print('->projections shape',projections.shape)


    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    # [7,3,4] * [7,4,160000] -> [7,3,480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    #print(f'->1.points in backproject inplace',points.shape)
    ## # # print('->feature shape',features.shape) # original: 6,64,225,400
    n_images, n_channels, height, width = features.shape # [7,256,97,97]
    #print(f'n_images:{n_images}')
    ## # # print('->0. points shape', points.shape) # [3,200,200,4]
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    #print(f'->2.points in backproject inplace',points.shape)

    ## # # print('->1. points shape', points.shape) #[3,200,200,4] -> [7,3,16000] -> []
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    #print(f'->3.points in backproject inplace',points.shape)

    ## # # print('2. points shape', points.shape) # [7,4,160000]
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]

    #print(f'->points shape in backproject in place',points.shape)
    #print(f'->projection shape in backproject in place',projection.shape)
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    #print('3. points_2d_3', points_2d_3.shape) # [7,3,160000]
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    #print(f'x shape,{x.shape}, y shape,{y.shape}, z shape, {z.shape}')
    # x shape,torch.Size([7, 160000]), y shape,torch.Size([7, 160000]), z shape, torch.Size([7, 160000])
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    #print('valid shape',valid.shape) # [7,160k]
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        # Method 2: Feature filling, only fill valid features, directly overwrite duplicate features
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    #print('penultimate volume', volume.shape) # [256,160000]
    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    #print('ultimate volume', volume.shape) # [256,200,200,4]

    # exit()
    return volume