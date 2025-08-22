# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 0, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks

class Muti_Decoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.decode_head1 = nn.Sequential(
            nn.Conv2d(48, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks,heatmap = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return masks, heatmap

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        if(sparse_prompt_embeddings is not None):
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        else:
            tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 0, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        #detection
        x = upscaled_embedding

        x = F.interpolate(
                        self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x =  self.decode_head2(x)
        return masks, x
    
class Muti_Dn_Decoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        query_num = 256
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.query_num = query_num

        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.Contrast_mlp = MLP(transformer_dim, transformer_dim, transformer_dim , 3)

        self.decode_head1 = nn.Sequential(
            nn.Conv2d(48, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dn_embeddings: torch.Tensor,
        dn_num
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks,heatmap,  feature = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dn_embeddings = dn_embeddings,
            dn_num = dn_num
        )

        # Prepare output
        return masks, heatmap,  feature

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dn_embeddings: torch.Tensor,
        dn_num
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        if(sparse_prompt_embeddings is not None):
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        else:
            tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = image_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        if dn_embeddings is None:
            attn_mask = None
            crossattn_mask = None
            hs, src = self.transformer(src, pos_src, tokens, attn_mask, crossattn_mask)
            mask_tokens_out = hs[:, 0, :]

            # Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(src)
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            #detection
            x = upscaled_embedding

            x = F.interpolate(
                            self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
            x =  self.decode_head2(x)
            return masks, x,  []
        else:
            # assert dn_embeddings.shape[1] % 2 == 0, "dn 数量错误"
            #prepare dn embeddings
            tokens , attn_mask, crossattn_mask, query_num, negative_num, n_prompt, positive_num = prepare_for_dn(tokens, dn_embeddings, src,query_num=self.query_num, backgroud_point=512)
            # assert n_prompt == 1, "dn 数量错误"

            # Run the transformer
            hs, src = self.transformer(src, pos_src, tokens, attn_mask, crossattn_mask)
            mask_tokens_out = hs[:, 0, :]

            # Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(src)
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            #detection
            x = upscaled_embedding

            x = F.interpolate(
                            self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
            x =  self.decode_head2(x)

            #Dn
            # dn_mask_tokens_out = torch.zeros([0, hs.shape[-1]]).to(hs.device)
            # dn_background_tokens_out = torch.zeros([0, hs.shape[-1]]).to(hs.device)
            # for i in range(group):
            #     dn_mask_tokens_out = torch.cat((dn_mask_tokens_out, hs[0, n_prompt + dn_point * i:n_prompt + dn_point * i + dn_point - 128, :]), dim=0)
            #     dn_background_tokens_out = torch.cat((dn_background_tokens_out, torch.mean(hs[:, n_prompt + dn_point * i + dn_point - 128:n_prompt + dn_point * (i + 1), :],dim = 1)), dim=0)
            # # dn_mask_tokens_out = hs[:, n_prompt:, :]
            # hyper_dn = self.output_hypernetworks_mlps(dn_mask_tokens_out).unsqueeze(0)
            # # masks = (hyper_dn @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
            # dn_masks = torch.einsum('bnc,bchw->bnhw', hyper_dn, upscaled_embedding).view(b, -1, h, w)

            # hyper_background = self.output_hypernetworks_mlps(dn_background_tokens_out).unsqueeze(0)
            # dn_background = torch.einsum('bnc,bchw->bnhw', hyper_background, upscaled_embedding).view(b, -1, h, w)

            #contrast
            feature_postive = torch.zeros([0, hs.shape[-1]]).to(hs.device)
            feature_query = torch.zeros([0, hs.shape[-1]]).to(hs.device)
            feature_negative = torch.zeros([0, hs.shape[-1]]).to(hs.device)

            feature = self.Contrast_mlp(hs[0,:, :])
            positive_num = dn_num[0]
            query_num = dn_num[1]
            negative_num = dn_num[2]

            assert n_prompt + positive_num + query_num + negative_num == feature.shape[0], "dn 数量错误"

            feature_postive = torch.cat((feature_postive, feature[n_prompt : n_prompt + positive_num, :]), dim=0)
            feature_query = torch.cat((feature_query, feature[n_prompt + positive_num :n_prompt + positive_num + query_num , :]), dim=0)
            feature_negative = torch.cat((feature_negative, feature[n_prompt + positive_num + query_num : , :]), dim=0)
            return masks, x, (feature_postive, feature_query, feature_negative)
# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

def prepare_for_dn( tokens, dn_embeddings, src, query_num = 256, backgroud_point = 512):
    """
    将tokens和dn_embeddings拼接,然后生成tokens看不到dn_embeddings并且dn_embeddings互相看不到的self-attention mask
    """
    _, n, d = dn_embeddings.shape#n为postive + query + negative数
    _, n_prompt, _ = tokens.shape#n_prompt为1+prompt数
    postive_num = n - query_num - backgroud_point
    tokens = torch.cat((tokens, dn_embeddings), dim=1)
    #初始化attention mask 全0表示可以看到
    attn_mask = torch.ones(tokens.size(1), tokens.size(1)).to(tokens.device) < 0
    #tokens看不到dn_embeddings
    attn_mask[:n_prompt,n_prompt: ] = True
    #dn_embeddings看不到prompt
    attn_mask[n_prompt:, :n_prompt] = True

    #生成cross attention mask
    crossattn_mask = torch.ones(src.size(-1) ** 2, tokens.size(1)).to(tokens.device) < 0
    crossattn_mask[:, n_prompt:] = True
    return tokens, attn_mask, crossattn_mask, query_num, backgroud_point, n_prompt, postive_num