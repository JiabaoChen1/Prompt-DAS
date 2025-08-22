# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .vision_transformer_dino import VisionTransformer
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder, PositionEmbeddingRandom

class Vit_seg_det(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: VisionTransformer,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        embedding_dim: int,
        vit_patch_size: int
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (DinoVisionTransformer): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.pe_layer = PositionEmbeddingRandom(embedding_dim // 2)
        self.patch_size = vit_patch_size

    def forward(
        self,
        input_images: torch.Tensor,
        point_coords: torch.Tensor,
        point_dn_coords : torch.Tensor = None,
        dn_num =[]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        B, C, H, W = input_images.shape
        image_embeddings = self.image_encoder(input_images)
        outputs = []
        outputs_det = []
        output_feature = []

        # outputs_det = self.det_decoder(image_embeddings)
        image_embeddings = image_embeddings.reshape([image_embeddings.shape[0], int(image_embeddings.shape[1]**0.5), int(image_embeddings.shape[1]**0.5), -1]).permute(0,3,1,2)
        if point_dn_coords is None:
          for point_coords, curr_embedding in zip(point_coords, image_embeddings):
              points = (point_coords.unsqueeze(0), torch.ones([point_coords.shape[0],1]).unsqueeze(0))
              sparse_embeddings, dn_embeddings = self.prompt_encoder(
                  points=points,
                  dn_points = None
              )
              low_res_masks,heatmap,_ = self.mask_decoder(
                  image_embeddings=curr_embedding.unsqueeze(0),
                  image_pe=self.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dn_embeddings = None,
                  dn_num = None
                  )
              low_res_masks = torch.sigmoid(low_res_masks)
              outputs.append(
                  low_res_masks
              )
              outputs_det.append(
                  heatmap
              )
          outputs = torch.stack(outputs).squeeze(1)
          outputs = outputs.repeat(1,2,1,1)
          outputs[:,0,::] = 1 - outputs[:,1,::]
          outputs_det = torch.stack(outputs_det).squeeze(1)
          return outputs, outputs_det
        else:
          for point_coords, point_dn_coords, curr_embedding, curr_num in zip(point_coords, point_dn_coords, image_embeddings, dn_num):
              points = (point_coords.unsqueeze(0), torch.ones([point_coords.shape[0],1]).unsqueeze(0))
              sparse_embeddings, dn_embeddings = self.prompt_encoder(
                  points=points,
                  dn_points = point_dn_coords
              )
              low_res_masks, heatmap, feature = self.mask_decoder(
                  image_embeddings=curr_embedding.unsqueeze(0),
                  image_pe=self.prompt_encoder.get_dense_pe(),
                  sparse_prompt_embeddings=sparse_embeddings,
                  dn_embeddings = dn_embeddings,
                  dn_num = curr_num
              )
              low_res_masks = torch.sigmoid(low_res_masks)
              outputs.append(
                  low_res_masks
              )
              outputs_det.append(
                  heatmap
              )
              output_feature.append(
                  feature
              )

          outputs = torch.stack(outputs).squeeze(1)
          outputs = outputs.repeat(1,2,1,1)
          outputs[:,0,::] = 1 - outputs[:,1,::]
          outputs_det = torch.stack(outputs_det).squeeze(1)
          return outputs, outputs_det,  output_feature