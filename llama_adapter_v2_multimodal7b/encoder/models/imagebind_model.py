#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from functools import partial
from types import SimpleNamespace
from collections import OrderedDict
import torch
import torch.nn as nn

from .helpers import (EinOpsRearrange, LearnableLogitScaling, Normalize,
                            SelectElement, SelectEOSAndProject)
from .multimodal_preprocessors import (SpatioTemporalPosEmbeddingHelper,
                                       PatchEmbedGeneric,
                                        TrajPreprocessor,
                                        EgoPreprocessor)
from .transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    TRAJ="traj",
    EGO="ego",
)

class ImageBindModel(nn.Module):
    def __init__(
        self,
        out_embed_dim=768,
        traj_embed_dim=512,
        traj_kernel_size=5,
        traj_num_blocks=6,
        traj_num_heads=8,
        traj_drop_path=0.7,
    ):
        super().__init__()

        self.modality_preprocessors = self._create_modality_preprocessors(
            traj_embed_dim
        )

        print(self.modality_preprocessors)
        self.modality_trunks = self._create_modality_trunks(
            traj_embed_dim,
            traj_num_blocks,
            traj_num_heads,
            traj_drop_path,
        )

        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            traj_embed_dim
        )

        self.modality_postprocessors = self._create_modality_postprocessors(
            out_embed_dim
        )
    
    def _create_modality_preprocessors(
        self,
        traj_embed_dim=512,
    ):
        
        traj_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=25, #25
                    out_features=traj_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=traj_embed_dim),
        )

        traj_preprocessor = TrajPreprocessor(
            img_size=[6, 5, 5],
            num_cls_tokens=1,
            kernel_size=5,
            embed_dim=traj_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            traj_stem=traj_stem,
        )

        ego_stem = PatchEmbedGeneric(
            [
                nn.Linear(
                    in_features=25, #25
                    out_features=traj_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=nn.LayerNorm(normalized_shape=traj_embed_dim),
        )

        ego_preprocessor = EgoPreprocessor(
            img_size=[1, 5, 5],
            num_cls_tokens=1,
            kernel_size=5,
            embed_dim=traj_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            traj_stem=ego_stem,
        )

        modality_preprocessors = {
            ModalityType.TRAJ: traj_preprocessor,
            ModalityType.EGO: ego_preprocessor,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_trunks(
        self,
        traj_embed_dim=512,
        traj_num_blocks=6,
        traj_num_heads=8,
        traj_drop_path=0.7,
    ):
        def instantiate_trunk(
            embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path
        ):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6)
                    if pre_transformer_ln
                    else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
            )

        modality_trunks = {}

        modality_trunks[ModalityType.TRAJ] = instantiate_trunk(
            traj_embed_dim,
            traj_num_blocks,
            traj_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=traj_drop_path,
        )

        modality_trunks[ModalityType.EGO] = instantiate_trunk(
            traj_embed_dim,
            traj_num_blocks,
            traj_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=traj_drop_path,
        )

        return nn.ModuleDict(modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        traj_embed_dim,
    ):
        modality_heads = {}

        modality_heads[ModalityType.TRAJ] = nn.Sequential(
            nn.LayerNorm(normalized_shape=traj_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(traj_embed_dim, out_embed_dim, bias=False),
        )
        modality_heads[ModalityType.EGO] = nn.Sequential(
            nn.LayerNorm(normalized_shape=traj_embed_dim, eps=1e-6),
            SelectElement(index=0),
            nn.Dropout(p=0.5),
            nn.Linear(traj_embed_dim, out_embed_dim, bias=False),
        )

        return nn.ModuleDict(modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):

        modality_postprocessors = {}

        modality_postprocessors[ModalityType.TRAJ] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.EGO] = nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        return nn.ModuleDict(modality_postprocessors)

    def forward(self, inputs, prenorm=False):
        outputs = {}
        outputs_prenorm = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            ) 
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                # print('after preprocessor:', modality_value)
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                
                modality_value = self.modality_heads[modality_key](
                    modality_value, **head_inputs
                )
               
                modality_value_postnorm = self.modality_postprocessors[modality_key](
                    modality_value
                )
                
                if reduce_list:
                    if prenorm:
                        modality_value = modality_value.reshape(B, S, -1)
                        modality_value = modality_value.mean(dim=1)

                    modality_value_postnorm = modality_value_postnorm.reshape(B, S, -1)
                    modality_value_postnorm = modality_value_postnorm.mean(dim=1)

                if prenorm:
                    outputs_prenorm[modality_key] = modality_value
                outputs[modality_key] = modality_value_postnorm

        if prenorm:
            return outputs, outputs_prenorm
        else:
            return outputs


def imagebind_huge():
    model = ImageBindModel(
        out_embed_dim=1024,
        traj_drop_path=0.7,
    )
    return model