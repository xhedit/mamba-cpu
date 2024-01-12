# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import os

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.config_mamba import MambaConfig
from mamba_ssm.mamba_simple import Mamba, Block
from mamba_ssm.generation import GenerationMixin
from mamba_ssm.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.layernorm import RMSNorm


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    layer_idx=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg)
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls)
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    layer_idx=i,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = RMSNorm(d_model, eps=norm_epsilon)

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(self, config: MambaConfig) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        rms_norm = config.rms_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        ssm_cfg = config.ssm_cfg

        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model
