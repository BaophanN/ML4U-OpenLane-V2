#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LaneSegNetDecoder(TransformerLayerSequence):

    def __init__(self, 
                 *args, 
                 prop_add_stage=0, # for what?
                 return_intermediate=False, 
                 pc_range=None, 
                 sample_idx=[0, 3, 6, 9], # num_ref_pts = len(sample_idx) * 2
                 pts_dim=3, 
                 **kwargs):
        super(LaneSegNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.sample_idx = sample_idx
        self.pts_dim = pts_dim
        self.prop_add_stage = prop_add_stage

    def forward(self,
                query,
                prop_query,   
                prop_reference_points, 
                *args,
                reference_points=None,
                reg_branches=None,
                cls_branches, 
                is_first_frame_list, 
                key_padding_mask=None,
                **kwargs):
        '''
            inter_states, inter_references = self.decoder(
            query=query, # (200,1,256)
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)
        '''

        output = query
        num_query, _, _ = output.shape
        # print('->Decoder, output shape', output.shape)
        intermediate = []
        intermediate_reference_points = []
        lane_ref_points = reference_points[:, :, self.sample_idx * 2, :]
        for lid, layer in enumerate(self.layers):

            # BS NUM_QUERY NUM_LEVEL NUM_REFPTS 3
            if lid == self.prop_add_stage and prop_query is not None and prop_reference_points is not None: 
                bs, topk, embed_dims = prop_query.shape 
                output = output.permute(1,0,2) 
                # print('->output',output)
                with torch.no_grad(): 
                    tmp_scores, _ = cls_branches[lid](output).max(-1)
                new_query = []
                new_refpts = [] 
                for i in range(bs): 
                    if is_first_frame_list[i]:
                        new_query.append(output[i])
                        new_refpts.append(reference_points[i])
                    else: 
                        _, valid_idx = torch.topk(tmp_scores[i],k=num_query - topk, dim=-1) 
                        new_query.append(torch.cat([prop_query[i],output[i][valid_idx]],dim=0))
                        new_refpts.append(torch.cat([prop_reference_points[i], reference_points[i][valid_idx]],dim=0))
                output = torch.stack(new_query).permute(1,0,2) 
                reference_points = torch.stack(new_refpts)
                assert list(output.shape) == [num_query, bs, embed_dims]

            reference_points_input = lane_ref_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                reg_center = reg_branches[0]
                reg_offset = reg_branches[1]

                tmp = reg_center[lid](output) # what is lid? 
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)

                assert reference_points.shape[-1] == self.pts_dim

                tmp = tmp + inverse_sigmoid(reference_points) # for what 
                tmp = tmp.sigmoid()
                reference_points = tmp.detach()

                coord = tmp.clone()
                coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1] # no idea  
                if self.pts_dim == 3:
                    coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                centerline = coord.view(bs, num_query, -1).contiguous()

                offset = reg_offset[lid](output)
                left_laneline = centerline + offset
                right_laneline = centerline - offset
                left_laneline = left_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]
                right_laneline = right_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]

                lane_ref_points = torch.cat([left_laneline, right_laneline], axis=-2).contiguous().detach()

                lane_ref_points[..., 0] = (lane_ref_points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                lane_ref_points[..., 1] = (lane_ref_points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                if self.pts_dim == 3:
                    lane_ref_points[..., 2] = (lane_ref_points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER.register_module()
class CustomDetrTransformerDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(CustomDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
    
    def forward(self,
                query,
                key=None,
                value=None,
                memory_query=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if memory_query is None:
                    temp_key = temp_value = query
                else:
                    temp_key = temp_value = torch.cat([memory_query, query], dim=0)
                
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
