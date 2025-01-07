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
        """
        6 images per time 
        """
        # #print('->intrinsic',results['cam_intrinsic'])
        # #print('->extrinsic',results['lidar2img'])
        projection = []
        for i in range(7):      
            intrinsic = torch.tensor(img_meta["cam_intrinsic"][i][:3, :3])
            intrinsic[:2] /= stride
            extrinsic = torch.tensor(img_meta["lidar2img"][i][:3])  
            if noise > 0:
                projection.append(intrinsic @ extrinsic + noise)
            else:
                projection.append(intrinsic @ extrinsic)
        return torch.stack(projection)
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        print('extract_img_feat, img',img.size(0), img.shape)
        # exit()
        if img is not None:
            # forward train 
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_() # remove the sequence dim 
            elif img.dim() == 5 and img.size(0) > 1: # has more than 1 image 
                B, N, C, H, W = img.size()
                # Post add 
                img = img.reshape(B * N, C, H, W)
                # img = img.reshape([-1] + list(img.shape)[2:])
                #print('->img reshape', img.shape) # 14,3, 800, 1024
                # [2,7,3,H,W] -> [7,3,H,W]
            # elif img.dim() == 4 and img.size(0) > 0: 
            #     N, C, H, W =  
            img_feats = self.img_backbone(img)
            # 2, 7,256,100,128 3 scale 
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            # _inner_forward
            img_feats = self.img_neck(img_feats)

        # img_feats_reshaped = [] # mlvl feats 
        img_feats_reshaped = [] 
        for img_feat in img_feats: 
            BN,C,H,W = img_feat.size() 
            if len_queue is not None: 
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else: 
                img_feats_reshaped.append(img_feat.view(B, int(BN/B),C,H,W))
        mlvl_feats = list(img_feats) # need this for creating features 3d
        features_2d = img_feats_reshaped # this for feeding the decoder
        #print('################')
        #### Post add 
        if self.multi_scale_id is not None: 
            # mlvl_feats: [1,128,100,128], [1,256,50,64], [1,512,25,32]
            # fusion with interpolation 
            mlvl_feats_ = []
            for msid in self.multi_scale_id: 
                # multi_scale_id: [0,1,2]
                if getattr(self, f'neck_fuse_{msid}', None) is not None: 
                    fuse_feats = [mlvl_feats[msid]]

                    for i in range(msid+1, len(mlvl_feats)): #  1->end 
                        resized_feat = resize(
                            mlvl_feats[i], 1
                            size=mlvl_feats[msid].size()[2:], # mlvl_feats[0]=[128,100,128] -> 
                            mode='bilinear',
                            align_corners=False
                        )
                        print(f'->resized feat {msid}, resize_feat.shape')
                        fuse_feats.append(resized_feat) 
                    if len(fuse_feats) > 1: 
                        fuse_feats = torch.cat(fuse_feats, dim=1) 
                    else: 
                        fuse_feats = fuse_feats[0]

                    print(f'fused feat, {fused_feats.shape}')    
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats) 
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_ 
        if isinstance(self.n_voxels,list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])
        
        mlvl_volumes = []
        print('->mlvl_feats', [mlvl_feat.shape for mlvl_feat in mlvl_feats])
        exit()
        for lvl, mlvl_feat in enumerate(mlvl_feats): 
            print(f'->mlvl_feat, {mlvl_feat.shape}') # 7,256,100,128
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # Calculate the stride
            print(f'->stride,img shape -1, mlvlfeat -1 {stride_i},{img.shape[-1]},{mlvl_feat.shape[-1]}')
            # [bs*nv, c, h, w] -> [bs, nv, c, h, w]
            #print('->mlvl_feat before reshape', mlvl_feat.shape) # [14,256,100,128], [b*nv,c,h,w]      

            mlvl_feat = mlvl_feat.reshape([B, -1] + list(mlvl_feat.shape[1:]))
            #print('->mlvl_feat after reshape', mlvl_feat.shape) # tensor[2,7,256,100,128], [b,nv,c,h,w]   
            #print('type mlvl_feat',type(mlvl_feat))
            # mlvl_feat_split = torch.split(mlvl_feat, 7, dim=0)
            # print('->mlvl_feat_split',len(mlvl_feat_split)) 
            # if batchsize=2, len=2          
            # print('->type mlvl_feat_split',type(mlvl_feat_split))
            # print('->mlvl_feat_split[0]', mlvl_feat_split[0].shape) # [2,7,256,100,128]
            volume_list = []
            # #print('->len',len(img_metas))
            for batch_id in range(mlvl_feat.shape[0]): # for bs 
                feats = []
                projection = []
                i = 0
                for view_id in range(mlvl_feat.shape[1]): # batchsize 2   # process by batch  
                    """
                    img_metas: [filename: 7, ori_shape: 7, crop_factor: 7]
                    """                  
                    # #print('batch id', batch_id)
                    # feat_i = mlvl_feat[batch_id][view_id] # N,C,H,W [bs][bs]? kind of wrong -> now use direct indexing 
                    # #print('mlvl_feat_split[batch_id] shape',mlvl_feat_split[batch_id].shape)
                    # #print('mlvl_feat_split shape', mlvl_feat_split[batch_id][batch_id].shape)
                    # #print('feat_i shape', feat_i.shape)
                    img_meta = img_metas[batch_id]
                    print(f'->batch_id,{batch_id},{img_meta}')
                    height = math.ceil(img_meta['img_shape'][view_id][0] / stride_i)
                    width = math.ceil(img_meta['img_shape'][view_id][1] / stride_i) 
                    print('-> height, width',height, width)
                    # exit()'
                    # compute projection         
                    intrinsic = torch.tensor(img_meta["cam_intrinsic"][view_id][:3, :3])
                    intrinsic[:2] /= stride_i
                    extrinsic = torch.tensor(img_meta["lidar2img"][view_id][:3])  
                    if self.extrinsic_noise > 0:
                        projection.append(intrinsic @ extrinsic + self.extrinsic_noise)
                    else:
                        projection.append(intrinsic @ extrinsic)
                    # projection = self._compute_projection(img_meta,stride_i,noise=self.extrinsic_noise).to(feat_i.device) 
                    # feats.append(feat_i) # 256,200,200,4
                    """
                        def __call__(self, results):
                    intrinsic = results['lidar2img']['intrinsic'][:3, :3]
                    extrinsic = results['lidar2img']['extrinsic'][0][:3, :3]
                    projection = intrinsic @ extrinsic
                    h, w, _ = results['ori_shape']
                    center_2d_3 = np.array([w / 2, h / 2, 1], dtype=np.float32)
                    center_2d_3 *= 3
                    origin = np.linalg.inv(projection) @ center_2d_3
                    results['lidar2img']['origin'] = origin
                    return results
                    """
                    print(f'->extract_img_feat, points shape, {points.shape}')
                        #print('volume vanilla',volume.shape)
                    # print('volume',volume.shape)
                if self.style in ['v1','v2']:
                    n_voxels, voxel_size = self.n_voxels[0],self.voxel_size[0]
                else: 
                    n_voxels, voxel_size = self.n_voxels[lvl],self.voxel_size[lvl]
                # volumes = torch.stack(volumes,dim=0) # [N,nx,ny,nz]
                projection = torch.stack(projection).to(mlvl_feat.device)
                points = get_points(
                    n_voxels=torch.tensor(n_voxels),
                    voxel_size=torch.tensor(voxel_size),
                    # where to get this origin 
                    origin=torch.tensor(img_meta['lidar2global_translation'])
                ).to(feat_i.device) 
                print('volume list[0]', volume_list[0].shape)
                if self.backproject == 'inplace': 
                    print('->feat i backproject inplace', feat_i.shape) # 2,7,256,100,128 
                    volumes = backproject_inplace(
                        # N,C,H,W 
                        mlvl_feat[batch_id], points, projection,
                    )
                    
                    #print('volume inplace', volume.shape) # [256,200,200,4]
                else: 
                    volumes, valid = backproject_vanilla(
                        mlvl_feat[batch_id],points,projection
                    )

                    volumes = volumes.sum(dim=0) 
                    valid = valid.sum(dim=0) 
                    volumes = volumes / valid 
                    valid = valid > 0 
                    volumes[:, ~valid[0]] = 0.0 
                print('-> volumes shape',volumes.shape)
            
                volume_list.append(torch.stack(volumes), dim=0) # [bs, c, vx,vy,vz]
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # Concatenate along channel dimension across batch [bs, nv*c, vx, vy, vz]
            print('mlvl_volumes[0]',mlvl_volumes[0].shape)
            if self.style in ['v1', 'v2']:
                # Concatenate directly along the channel dimension for multi-levels without sequence
                mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*nv*c, vx, vy, vz]
            else:
                # bev ms: multi-scale bev map (different x/y/z)
                for i in range(len(mlvl_volumes)):
                    mlvl_volume = mlvl_volumes[i]
                    #print('1. mlvl_volume shape', mlvl_volume.shape) # [1,256,200,200,4]
                    bs, c, x, y, z = mlvl_volume.shape
                    # 2,256,200,200,4
                    # Collapse z-axis, reshaping [bs, nv*c, vx, vy, vz] -> [bs, nv*c*vz, vx, vy]
                    mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z * c).permute(0, 3, 1, 2)
                    #print('2. mlvl_volume shape', mlvl_volume.shape) # [1,1024,200,200]
                    # [2,256,200,200,4]->[2,200,200,4,256]->[2,1024,200,200]
                    # Align x and y dimensions based on multi-scale configuration
                    if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                        # Downscale to match the resolution of the lowest scale
                        mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                    elif self.multi_scale_3d_scaler == 'upsample' and i != 0:
                        # Upscale to match the resolution of the highest scale
                        mlvl_volume = resize(
                            mlvl_volume,
                            mlvl_volumes[0].size()[2:4],
                            mode='bilinear',
                            align_corners=False)
                    # No change if it already matches the target resolution
                    # [2,1024,200,200,1]F
                    # Expand to [bs, nv*c*vz, vx, vy, 1] to add a placeholder for z-axis consistency
                    # 
                    mlvl_volume = mlvl_volume.unsqueeze(-1)
                    #print('3. mlvl_volume after unsqueeze', mlvl_volume.shape)# [1,1024,200,200,1]

                    mlvl_volumes[i] = mlvl_volume

                # Concatenate all multi-scale levels along the channel dimension
                mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, sum(z*c) across levels, vx, vy, 1]
                # B,1024,200,200,1
                #print('4. mlvl_volumes shape', mlvl_volumes.shape)
                

        x = mlvl_volumes
        x = self.neck_3d(x) # [1,256,100,100]
        #print('->bev feats type: ',type(x)) # no multiscale 
        #print('len bev feats', len(x))
        #print('enumerate shape of bev feats',[y.shape for y in x])
        #print('->bev feat shape', torch.stack(x).shape)
        # [1,192,100,100]
        #print('->features 2d type:',type(features_2d))
        #print('enumerate shape of features 2d', [y.shape for y in features_2d])
        #print('->len feature 2d', len(features_2d)) # 4 scale from resnet+FPN
        # import sys;sys.exit()
        # #print('->features 2d type:',torch.stack(features_2d).shape)
        """
        
        Output has
        features 2d: [7,256,100,128], [7,256,50,64], [7,256,25,32], [7,256,13,16]
        mlvl_feats: [7,256, 100, 128]
        """
        return x, None, features_2d
        # bev_feats, valids, img_feats 
        #### for later check up
        # for img_feat in img_feats:
        #     BN, C, H, W = img_feat.size()
        #     if len_queue is not None:
        #         img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
        #     else:
        #         img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        # return img_feats_reshaped

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
        print('->forward train, img shape', img.shape)
        print('->forward_train. len img_metas', len(img_metas))
        print('->forward train, img metas[0]', len(img_metas[0]))
        # print(img_metas[0][1]['filename'])
        # print(img_metas[0]['ori_shape'])
        #sys.exit()
        # print(f'img_metas: {img_metas}')
        # B,N,C,H,W  
        prev_img = img[:, :-1, ...] # 
        img = img[:, -1, ...]

        if self.video_test_mode:
            prev_img_metas = copy.deepcopy(img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        bev_feats, valids, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # this is bev_feats right now 

        # bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        # print(f'len img_metas, {len(img_metas)}')
        losses = dict()
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
        # print(f'forward_test->len img, {len(img)}')
        # print(f'->img[0] shape:{img[0].shape}') # 7,3,800,1024
        # print(f'->len img_metas:{len(img_metas)}') # 
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
        print('->forward test, img shape', img.shape)
        print('->forward test, img metas cak', len(img_metas[0]))
        print(img_metas[0]['ori_shape'])
        # sys.exit()
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
        _, _, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        """
        extract_feat(self, img, img_metas=None, len_queue=None) returns 
        """

        """
        bev_feats, valids, img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return bev_feats, valids, img_feats
        """
        results_list = [dict() for i in range(len(img_metas))]
        # print(f'->img_metas: {img_metas}')
        new_prev_bev, lane_results, lsls_results = self.simple_test_pts(
            img_feats, img_metas, img, prev_bev, rescale=rescale)
        #   x,       , 1 set of 7 imgs, 
        for result_dict, lane, lsls in zip(results_list, lane_results, lsls_results):
            result_dict['lane_results'] = lane
            result_dict['bbox_results'] = None
            result_dict['lsls_results'] = lsls
            result_dict['lste_results'] = None

        return new_prev_bev, results_list
    def simple_test_pts(self, x, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function"""
        batchsize = len(img_metas)
        # print('->batchsize len', batchsize)
        # import sys;sys.exit()

        bev_feats, _, img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # bev_feats = self.bev_constructor(x, img_metas, prev_bev)
        # outs = self.pts_bbox_head(mlvl_feats=img_feats, bev_feats=bev_feats, img_metas=img_metas)

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



@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    """
    n_voxels: [200,200,4]
    voxel_size: 0.5,0.5,1.5
    origin: 
    """
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )

    new_origin = origin - n_voxels / 2.0 * voxel_size
    # 200,200,4 x 
    # print(f'get_points, points, {points.shape}')
    # print(f'get_points, voxel_size, {voxel_size.shape}')
    # print(f'get_points, new_origin, {new_origin.shape}')

    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
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
    #print('->feature shape',features.shape)
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
    #print('->feature shape',features.shape) # original: 6,64,225,400
    n_images, n_channels, height, width = features.shape # [7,256,97,97]
    #print('->0. points shape', points.shape) # [3,200,200,4]
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    #print('->1. points shape', points.shape) #[3,200,200,4] -> [7,3,16000] -> []
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    #print('2. points shape', points.shape) # [7,4,160000]
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    # print(f"->projection shape: {projection.shape}")
    # print(f"->points shape: {points.shape}, invoke:{invoke_time}")

    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    #print('3. points_2d_3', points_2d_3.shape) # [7,3,160000]
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    #print(f'x shape,{x.shape}, y shape,{y.shape}, z shape, {z.shape}')
    # x shape,torch.Size([7, 160000]), y shape,torch.Size([7, 160000]), z shape, torch.Size([7, 160000])
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    #print('valid shape',valid.shape) # [7,160k]
    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    #print('penultimate volume', volume.shape) # [256,160000]
    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    #print('ultimate volume', volume.shape) # [256,200,200,4]
    return volume