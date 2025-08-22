import torch

from functools import partial

from .modeling import (
    PromptEncoder,
    TwoWayDnTransformer,
    Vit_seg_det,
    Muti_Dn_Decoder,
)
from .modeling.vision_transformer_dino import vit_small, vit_base

def build_mutitask_dn(backbone="vit_s", checkpoint=None, patch_size=8, image_size=256, query_num=256):
    prompt_embed_dim = 384
    image_size = image_size
    vit_patch_size = patch_size
    image_embedding_size = image_size // vit_patch_size
    if backbone == "vit_s":
        image_encoder = vit_small(patch_size=vit_patch_size, checkpoint=checkpoint)
        prompt_embed_dim = 384
    elif backbone == "vit_b":
        image_encoder = vit_base(patch_size=vit_patch_size)
        prompt_embed_dim = 768

    vit_seg = Vit_seg_det(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
        ),
        mask_decoder=Muti_Dn_Decoder(
            transformer=TwoWayDnTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            query_num=query_num,
        ),
        embedding_dim=prompt_embed_dim,
        vit_patch_size=vit_patch_size,
    )
    vit_seg.eval()

    return vit_seg
