import os
from functools import partial
import torch

from .vmamba import VSSM


def build_vssm_model(config, **kwargs):
    model_type = config.type
    # Error in this? the object is not built successfully
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.patch_size, 
            in_chans=config.in_chans, 
            num_classes=config.num_classes, 
            depths=config.depths, 
            dims=config.dims, 
            # ===================
            ssm_d_state=config.ssm_d_state,
            ssm_ratio=config.ssm_ratio,
            ssm_rank_ratio=config.ssm_rank_ratio,
            ssm_dt_rank=("auto" if config.ssm_dt_rank == "auto" else int(config.ssm_dt_rank)),
            ssm_act_layer=config.ssm_act_layer,
            ssm_conv=config.MODEL.ssm_conv,
            ssm_conv_bias=config.ssm_conv_bias,
            ssm_drop_rate=config.ssm_drop_rate,
            ssm_init=config.ssm_init,
            forward_type=config.ssm_forward_type,
            # ===================
            mlp_ratio=config.mlp_ratio,
            mlp_act_layer=config.mlp_act_layer,
            mlp_drop_rate=config.mlp_drop_rate,
            # ===================
            drop_path_rate=config.drop_path_rate,
            patch_norm=config.patch_norm,
            norm_layer=config.norm_layer,
            downsample_version=config.downsample,
            patchembed_version=config.patch_embed,
            gmlp=config.gmlp,
            use_checkpoint=config.use_checkpoint,
            # ===================
            posembed=config.posembed,
            imgsize=config.imgsize,
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import simple_build
        model = simple_build(config.MODEL.TYPE)
    return model


if __name__ == "__main__": 
    img_backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        norm_layer="ln2d",
    )
    model = build_vssm_model(img_backbone)
    print(model)

