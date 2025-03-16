#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import html
import io
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from timm.models.layers import trunc_normal_

from .helpers import VerboseNNModule, cast_if_src_dtype


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    # nn.functional.interpolate doesn't work with bfloat16 so we cast to float32
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch.float32)
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
            0, 3, 1, 2
        ),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed


def interpolate_pos_encoding(
    npatch_per_img,
    pos_embed,
    patches_layout,
    input_shape=None,
    first_patch_idx=1,
):
    assert first_patch_idx == 0 or first_patch_idx == 1, "there is 1 CLS token or none"
    N = pos_embed.shape[1] - first_patch_idx  # since it's 1 if cls_token exists
    if npatch_per_img == N:
        return pos_embed

    assert (
        patches_layout[-1] == patches_layout[-2]
    ), "Interpolation of pos embed not supported for non-square layouts"

    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]

    if input_shape is None or patches_layout[0] == 1:
        # simple 2D pos embedding, no temporal component
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
    elif patches_layout[0] > 1:
        # pos embed has a temporal component
        assert len(input_shape) == 4, "temporal interpolation not supported"
        # we only support 2D interpolation in this case
        num_frames = patches_layout[0]
        num_spatial_tokens = patches_layout[1] * patches_layout[2]
        pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
        # interpolate embedding for zeroth frame
        pos_embed = interpolate_pos_encoding_2d(
            npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0)
        )
    else:
        raise ValueError("This type of interpolation isn't implemented")

    return torch.cat((class_emb, pos_embed), dim=1)


def _get_pos_embedding(
    npatch_per_img,
    pos_embed,
    patches_layout,
    input_shape,
    first_patch_idx=1,
):
    pos_embed = interpolate_pos_encoding(
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=input_shape,
        first_patch_idx=first_patch_idx,
    )
    return pos_embed


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem, norm_layer: Optional[nn.Module] = None):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            dummy_out = self.proj(dummy_img)
        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)
        return patches_layout, num_patches, embed_dim

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        x = x.flatten(2).transpose(1, 2)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class SpatioTemporalPosEmbeddingHelper(VerboseNNModule):
    def __init__(
        self,
        patches_layout: List,
        num_patches: int,
        num_cls_tokens: int,
        embed_dim: int,
        learnable: bool,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.patches_layout = patches_layout
        self.num_patches = num_patches
        self.num_tokens = num_cls_tokens + num_patches
        self.learnable = learnable
        if self.learnable:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed", get_sinusoid_encoding_table(self.num_tokens, embed_dim)
            )

    def get_pos_embedding(self, vision_input, all_vision_tokens):
        input_shape = vision_input.shape
        pos_embed = _get_pos_embedding(
            all_vision_tokens.size(1) - self.num_cls_tokens,
            pos_embed=self.pos_embed,
            patches_layout=self.patches_layout,
            input_shape=input_shape,
            first_patch_idx=self.num_cls_tokens,
        )
        return pos_embed


class TrajPreprocessor(VerboseNNModule):
    def __init__(
        self,
        kernel_size: int,
        traj_stem: PatchEmbedGeneric,
        embed_dim: int,
        img_size: Tuple = (6, 5, 5),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.traj_stem = traj_stem
        self.embed_dim = embed_dim
        self.use_pos_embed = pos_embed_fn is not None
        self.num_cls_tokens = num_cls_tokens
        self.kernel_size = kernel_size
        self.pos_embed = nn.Parameter(
            torch.empty(1, (img_size[0]) + num_cls_tokens, embed_dim)
        )

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5

            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def tokenize_input_and_cls_pos(self, input, stem):
        # tokens is of shape B x L x D
        tokens = stem.norm_layer(stem.proj(input))
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        return tokens

    def forward(self, traj):
        # Patchify
        traj = traj.unfold(
            -1,
            self.kernel_size,
            self.kernel_size,
        ).permute(0, 1, 3, 2, 4)
        traj = traj.reshape(traj.size(0), traj.size(1), -1)

        traj_tokens = self.tokenize_input_and_cls_pos(
            traj,
            self.traj_stem,
        )

        return_dict = {
            "trunk": {
                "tokens": traj_tokens,
            },
            "head": {},
        }
        return return_dict
    
class EgoPreprocessor(VerboseNNModule):
    def __init__(
        self,
        kernel_size: int,
        traj_stem: PatchEmbedGeneric,
        embed_dim: int,
        img_size: Tuple = (1, 5, 5),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.traj_stem = traj_stem
        self.embed_dim = embed_dim
        self.use_pos_embed = pos_embed_fn is not None
        self.num_cls_tokens = num_cls_tokens
        self.kernel_size = kernel_size
        self.pos_embed = nn.Parameter(
            torch.empty(1, (img_size[0]) + num_cls_tokens, embed_dim)
        )

        if self.num_cls_tokens > 0:
            self.cls_token = nn.Parameter(
                torch.zeros(1, self.num_cls_tokens, self.embed_dim)
            )

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5

            if self.num_cls_tokens > 0:
                nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == "vit":
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def tokenize_input_and_cls_pos(self, input, stem):
        # tokens is of shape B x L x D
        tokens = stem.norm_layer(stem.proj(input))
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        return tokens

    def forward(self, ego):
        # Patchify
        traj = ego.unfold(
            -1,
            self.kernel_size,
            self.kernel_size,
        ).permute(0, 1, 3, 2, 4)
        traj = traj.reshape(traj.size(0), traj.size(1), -1)

        traj_tokens = self.tokenize_input_and_cls_pos(
            traj,
            self.traj_stem,
        )

        return_dict = {
            "trunk": {
                "tokens": traj_tokens,
            },
            "head": {},
        }
        return return_dict
