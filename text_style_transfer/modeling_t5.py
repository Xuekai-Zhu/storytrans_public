# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """

import copy
import math
import os
import warnings
from typing import Optional
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from transformers.activations import ACT2FN

from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_utils import top_k_top_p_filtering
from transformers.generation_logits_process import LogitsWarper


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:
                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24
    Example::
            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],
                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example::
        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],
                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class Pointer_Network(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        # 这里固定了只有一层
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(1)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # pointer network 计算指针attention
        # TODO：计算指针并添加
        
        pointing_score = torch.matmul(hidden_states, inputs_embeds.permute(0,2,1)) / hidden_states.size(-1)
        
        
        return pointing_score
        
        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [
        #             hidden_states,
        #             present_key_value_states,
        #             all_hidden_states,
        #             all_attentions,
        #             all_cross_attentions,
        #         ]
        #         if v is not None
        #     )
        # return BaseModelOutputWithPastAndCrossAttentions(
            # last_hidden_state=hidden_states,
            # past_key_values=present_key_value_states,
            # hidden_states=all_hidden_states,
            # attentions=all_attentions,
            # cross_attentions=all_cross_attentions,)


class T5Stack_add_sen_type(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, sen_type_emb=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.embed_sens = sen_type_emb
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        input_sentence_types=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        # 增加sen type
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            # 去掉sen type
            # inputs_sens_embeds = self.embed_sens(input_sentence_types)
            # inputs_embeds = inputs_sens_embeds + inputs_sens_embeds

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""
    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.
            `What are input IDs? <../glossary.html#input-ids>`__
            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__
            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in ``[0,
            1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.
            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5Model
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs



# my model


class T5EncoderModel_AddSoftSampling(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class Style_Classifier(nn.Module):
    def __init__(self, config):
        super(Style_Classifier, self).__init__()

        self.encoder = T5EncoderModel_AddSoftSampling.from_pretrained(config.pre_trained_t5)

        self.linear = nn.Linear(config.in_size, config.style_num)

    def forward(self, ids, soft_sampling=False):

        if soft_sampling is True:
            # freeze
            for p in self.parameters():
                p.requires_grad = False

            ids = torch.matmul(ids, self.encoder.shared.weight)
            outputs = self.encoder(inputs_embeds=ids)
        else:
            outputs = self.encoder(input_ids=ids)
        last_hidden_states = outputs.last_hidden_state
        batch_respresentation = torch.mean(last_hidden_states, dim=1)
        pred = self.linear(batch_respresentation)

        return pred


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForAddStyleEmb(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # style embedding
        # self.style_emb = nn.Embedding(style_model_config.style_num, 768)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        soft_sampling=False,
        raw_style_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        # do soft sampling
        if inputs_embeds is not None and soft_sampling is not False and raw_style_ids is not None:
            inputs_embeds = torch.matmul(inputs_embeds, self.shared.weight)
            raw_style_embs = self.shared(raw_style_ids)
            inputs_embeds = torch.cat((raw_style_embs.unsqueeze(1), inputs_embeds), 1)[:, :self.config.max_length, :]

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(reduction="sum", ignore_index=-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class StyleContentInter(T5PreTrainedModel):
    def __init__(self, config, num_layers=None, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class LongTextSTEncoderAndInter(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.project_content_768 = nn.Linear(384, 768)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output):
        sen_id = 32003
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        return content_representation, style_re, posi[0]

    def style_content_interaction_module(self, trans_style_emb, content_representation, posi):
        # prepare index
        pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        batch = trans_style_emb.size(0)
        index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # tensor pool
        content_representation = self.transfer_size_for_content(content_representation)
        tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # index gather
        batch_sen_hidden = tensor_pool[index_list]
        sen_ids = index_list.ne(-1).long()
        sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=batch_sen_hidden_add_style)

        return hidden.last_hidden_state, sen_mask

    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            content_representation, style_representation, posi = self.get_sen_representation(input_ids, hidden_states)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            sen_hidden, sen_mask = self.style_content_interaction_module(trans_to_style_emb, content_representation, posi)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )




@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class LongTextST(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class StyleTransOutput(object):
    def __init__(self,
                 loss=None,
                 logits=None,
                 past_key_values=None,
                 decoder_hidden_states=None,
                 decoder_attentions=None,
                 cross_attentions=None,
                 encoder_last_hidden_state=None,
                 encoder_hidden_states=None,
                 encoder_attentions=None,
                 style_representation=None,
                 content_representation=None,
                 batch_content=None,
                 content_label=None,
                 pointing_res=None,):

        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.decoder_hidden_states = decoder_hidden_states
        self.decoder_attentions = decoder_attentions
        self.cross_attentions = cross_attentions
        self.encoder_last_hidden_state = encoder_last_hidden_state
        self.encoder_hidden_states = encoder_hidden_states
        self.encoder_attentions = encoder_attentions
        self.style_representation = style_representation
        self.content_representation = content_representation
        self.batch_content = batch_content
        self.content_label = content_label
        self.pointing_res = pointing_res

class LongTextSTEncoderAndInter_Output(object):
    def __init__(self,
                 last_hidden_state=None,
                 hidden_states=None,
                 attentions=None,
                 style_representation=None,
                 content_representation=None,
                 sen_hidden=None,
                 sen_mask=None,
                 batch_content=None,
                 trasfered_representation=None,
                 content_label=None,
                 ):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.style_representation = style_representation
        self.content_representation = content_representation
        self.sen_hidden = sen_hidden
        self.sen_mask = sen_mask
        self.batch_content = batch_content
        self.trasfered_representation = trasfered_representation
        self.content_label=content_label

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(None, logits)

    return logits

class TopPLogitsWarper(LogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.

    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class LongTextSTEncoderAndInter_Disentangled(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output):
        sen_id = 32003
        batch = input_ids.size(0)
        sen_posi = torch.where(input_ids == sen_id)
        sen_tensors = encoder_output[sen_posi]
        end_posi = sen_posi[1]
        # start_posi = torch.cat((torch.zeros(1, dtype=torch.int).to(end_posi.device), end_posi+1), dim=0)[:-1]
        sen_list = []
        tokens_mean_list = []
        for i in range(batch):
            single_sen = torch.mean(sen_tensors[sen_posi[0] == i], dim=0, keepdim=True)
            sen_list.append(single_sen)
            end = end_posi[sen_posi[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # a = encoder_output[i, m:n, :]
                sentence = torch.mean(encoder_output[i, m:n, :], dim=0, keepdim=True)
                tokens_mean_list.append(sentence)

        batch_style = torch.cat(sen_list, dim=0)
        batch_sentence = torch.cat(tokens_mean_list, dim=0)

        return batch_style, batch_sentence, sen_posi

    def style_content_interaction_module(self, trans_style_emb, hidden_states, posi):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        batch = trans_style_emb.size(0)
        batch_posi = posi[0]
        sen_posi = posi[1]
        for i in range(batch):
            a = sen_posi[batch_posi == i]
            hidden_states[i, a] = trans_style_emb[i]
        hidden = self.interaction_module(inputs_embeds=hidden_states)

        return hidden.last_hidden_state

    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            batch_style, batch_sentence, sen_posi = self.get_sen_representation(input_ids, hidden_states)
            # style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            new_hidden = self.style_content_interaction_module(trans_to_style_emb, hidden_states, sen_posi)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=batch_style,
                content_representation=batch_sentence,
                trasfered_representation=new_hidden,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )





class LongTextST_Disentangled(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Disentangled(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
            )

        # hidden_states = encoder_outputs[0]
        # sen_hidden = encoder_outputs.sen_hidden
        hidden_states = encoder_outputs.trasfered_representation

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            # batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class LongTextSTEncoderAndInter_Concat(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        # self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output):
        sen_id = 32003
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, content_re, style_re, posi[0]

    def style_content_interaction_module(self, trans_style_emb, content_representation, posi):
        # prepare index
        pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        batch = trans_style_emb.size(0)
        index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # tensor pool
        content_representation = self.transfer_size_for_content(content_representation)
        tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # index gather
        batch_sen_hidden = tensor_pool[index_list]
        sen_ids = index_list.ne(-1).long()
        sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=batch_sen_hidden_add_style)

        return hidden.last_hidden_state, sen_mask

    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # sen_hidden, sen_mask = self.style_content_interaction_module(trans_to_style_emb, content_representation, posi)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Concat(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Concat(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]

class LongTextSTEncoderAndInter_Concat_Inter(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output):
        sen_id = 32003
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, content_re, style_re, posi[0]

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(new_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Concat_Inter(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Concat_Inter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class LongTextSTEncoderAndInter_Disturb(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        # self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, content_re, style_re, posi[0]

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, bt_content, hidden, input_ids, sen_id):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], bt_content[sen_num])
                score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        new_hidden = torch.mul(hidden, score_mat_add_eos.unsqueeze(-1))
        return new_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens_aff, content)
        cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        print(cos[cos>0.7])
        print(matmul[matmul>0.7])
        attention_score_v2 = F.normalize(matmul.unsqueeze())
        attention_score = F.softmax(matmul, dim=-1)
        print(attention_score[attention_score>0.7])
        print(attention_score_v2[attention_score_v2>0.7])

        return cos



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
        project_linear=None):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is not None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # inter_hidden = self.style_content_interaction_module(new_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )
        elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
                                                                                                            hidden_states,
                                                                                                            sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            pred_bt_sen_representation = project_linear(content_representation)
            disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Disturb(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Disturb(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id, project_linear):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None, project_linear=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id, project_linear)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class LongTextSTEncoderAndInter_Attention(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, content_re, style_re, posi[0]

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, bt_content, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], bt_content[sen_num])
                score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        return new_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)

        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens_aff, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        attention_score = F.sigmoid(matmul)
        # a = attention_score.detach().cpu().numpy()

        return attention_score



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
                ):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is not None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )
        elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
                                                                                                            hidden_states,
                                                                                                            sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            pred_bt_sen_representation = self.project_content_768(content_representation)
            disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Attention(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Attention(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                # project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class LongTextSTEncoderAndInter_Test(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, content_re, style_re, posi[0]

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, bt_content, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], bt_content[sen_num])
                score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        return new_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)

        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens_aff, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        attention_score = F.sigmoid(matmul)
        # a = attention_score.detach().cpu().numpy()

        return attention_score



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
                ):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is not None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            # inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                # sen_hidden=inter_hidden,
                sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )
        elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
            assert sen_id != None, "sen_id is None"
            content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
                                                                                                            hidden_states,
                                                                                                            sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # pred_bt_sen_representation = self.project_content_768(content_representation)
            # disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            # inter_hidden = self.style_content_interaction_module(disturb_hidden)
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Test(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Test(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                # project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class LongTextSTEncoderAndInter_Style(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        return content_representation, style_re, posi[0]

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self,bt_content, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], bt_content[sen_num])
                score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        return new_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        # transfer_hidden = hidden.scatter(dim=1, )
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens_aff, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        attention_score = F.sigmoid(matmul)
        # a = attention_score.detach().cpu().numpy()

        return attention_score



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
                ):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is not None:
            assert sen_id != None, "sen_id is None"
            content_representation, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )
        elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
            assert sen_id != None, "sen_id is None"
            content_representation, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            pred_bt_sen_representation = self.project_content_768(content_representation)
            disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Style(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Style(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                # project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]

class LongTextSTEncoderAndInter_Style_Attention(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        # style_num = 3
        style_num = 2
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(sen_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        return style_re, sen_representation

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, sen_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], sen_representation[sen_num])
                score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        return new_hidden
        # return trans_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        # tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        attention_score = F.softmax(matmul, dim=-1)
        reverse_score = 1 - attention_score
        # a = reverse_score.detach().cpu().numpy()
        # print(a)

        return reverse_score



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
                ):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            assert sen_id != None, "sen_id is None"
            # content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation, sen_representation = self.get_sen_representation(input_ids, hidden_states, sen_id)
            # style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            disturb_hidden = self.disturb(sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                # content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )
        # elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
        #     assert sen_id != None, "sen_id is None"
        #     content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
        #                                                                                                     hidden_states,
        #                                                                                                     sen_id)
        #     style_representation = self.transfer_size_for_style(style_representation)
        #     trans_to_style_emb = self.style_embedding(transfer_to)
        #     pred_bt_sen_representation = self.project_content_768(content_representation)
        #     disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        #     # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
        #     inter_hidden = self.style_content_interaction_module(disturb_hidden)
        #     return LongTextSTEncoderAndInter_Output(
        #         last_hidden_state=encoder_outputs.last_hidden_state,
        #         # hidden_states=encoder_outputs.hidden_states,
        #         # attentions=encoder_outputs.attentions,
        #         style_representation=style_representation,
        #         content_representation=content_representation,
        #         sen_hidden=inter_hidden,
        #         # sen_mask=sen_mask,
        #         batch_content=batch_content,
        #     )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Style_Attention(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Style_Attention(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                # project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]

class LongTextSTEncoderAndInter_Style_Change(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(sen_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)


        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        return style_re, sen_representation

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, sen_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        # batch = input_ids.size(0)
        # sen_num = 0
        # all_score = []
        # for i in range(batch):
        #     batch_score = []
        #     end = sen_index[1][sen_index[0] == i]
        #     start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
        #     for m, n in zip(start, end):
        #         score = self.sim_content_tokens(hidden[i, m:n], sen_representation[sen_num])
        #         score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
        #         batch_score.append(score_sen)
        #         sen_num += 1
        #     single_batch_score = torch.cat((batch_score), dim=-1)
        #     all_score.append(single_batch_score)
        # score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        # new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        # return new_hidden
        return trans_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        # tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        attention_score = F.softmax(matmul, dim=-1)
        reverse_score = 1 - attention_score
        # a = reverse_score.detach().cpu().numpy()
        # print(a)

        return reverse_score



    def forward(self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        bt_sen_representation=None,
        sen_id=None,
                ):

        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]


        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            assert sen_id != None, "sen_id is None"
            # content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation, sen_representation = self.get_sen_representation(input_ids, hidden_states, sen_id)
            # style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            disturb_hidden = self.disturb(sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                # content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )
        # elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
        #     assert sen_id != None, "sen_id is None"
        #     content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
        #                                                                                                     hidden_states,
        #                                                                                                     sen_id)
        #     style_representation = self.transfer_size_for_style(style_representation)
        #     trans_to_style_emb = self.style_embedding(transfer_to)
        #     pred_bt_sen_representation = self.project_content_768(content_representation)
        #     disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        #     # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
        #     inter_hidden = self.style_content_interaction_module(disturb_hidden)
        #     return LongTextSTEncoderAndInter_Output(
        #         last_hidden_state=encoder_outputs.last_hidden_state,
        #         # hidden_states=encoder_outputs.hidden_states,
        #         # attentions=encoder_outputs.attentions,
        #         style_representation=style_representation,
        #         content_representation=content_representation,
        #         sen_hidden=inter_hidden,
        #         # sen_mask=sen_mask,
        #         batch_content=batch_content,
        #     )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class LongTextST_Style_Change(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Style_Change(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_hidden=None,
        bt_sen_representation=None,
        sen_id=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                return_dict=True,
                transfer_to=transfer_to,
                sen_id=sen_id,
                # project_linear=project_linear,
            )

        return encoder_outputs


    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None, temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class LongTextSTEncoderAndInter_Content_Dis(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        self.attention_project = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(sen_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)

        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        return style_re, sen_representation

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, sen_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        # batch = input_ids.size(0)
        # sen_num = 0
        # all_score = []
        # for i in range(batch):
        #     batch_score = []
        #     end = sen_index[1][sen_index[0] == i]
        #     start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
        #     for m, n in zip(start, end):
        #         score = self.sim_content_tokens(hidden[i, m:n], sen_representation[sen_num])
        #         score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
        #         batch_score.append(score_sen)
        #         sen_num += 1
        #     single_batch_score = torch.cat((batch_score), dim=-1)
        #     all_score.append(single_batch_score)
        # score_mat = pad_sequence(all_score, batch_first=True, padding_value=1)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        trans_hidden = self.transfer_hidden(sen_index, hidden, transfer_to_emb, input_ids)
        # new_hidden = torch.mul(trans_hidden, score_mat_add_eos.unsqueeze(-1))
        # return new_hidden
        return trans_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        # tokens_aff = self.attention_project(tokens)
        matmul = torch.matmul(tokens, content)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        attention_score = F.softmax(matmul, dim=-1)
        reverse_score = 1 - attention_score
        # a = reverse_score.detach().cpu().numpy()
        # print(a)

        return reverse_score

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                transfer_to=None,
                bt_sen_representation=None,
                sen_id=None,
                ):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            assert sen_id != None, "sen_id is None"
            # content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation, sen_representation = self.get_sen_representation(input_ids, hidden_states,
                                                                                   sen_id)
            # style_representation = self.transfer_size_for_style(style_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            disturb_hidden = self.disturb(sen_representation, hidden_states, input_ids, sen_id,
                                          trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                # content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )
        # elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
        #     assert sen_id != None, "sen_id is None"
        #     content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
        #                                                                                                     hidden_states,
        #                                                                                                     sen_id)
        #     style_representation = self.transfer_size_for_style(style_representation)
        #     trans_to_style_emb = self.style_embedding(transfer_to)
        #     pred_bt_sen_representation = self.project_content_768(content_representation)
        #     disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        #     # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
        #     inter_hidden = self.style_content_interaction_module(disturb_hidden)
        #     return LongTextSTEncoderAndInter_Output(
        #         last_hidden_state=encoder_outputs.last_hidden_state,
        #         # hidden_states=encoder_outputs.hidden_states,
        #         # attentions=encoder_outputs.attentions,
        #         style_representation=style_representation,
        #         content_representation=content_representation,
        #         sen_hidden=inter_hidden,
        #         # sen_mask=sen_mask,
        #         batch_content=batch_content,
        #     )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )



class LongTextST_Content_Dis(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Content_Dis(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        self.noraml_mean_weight = torch.nn.Parameter(torch.zeros(config.d_model, requires_grad=True))
        self.noraml_covariance_matrix = torch.nn.Parameter(torch.ones(config.d_model, requires_grad=True))
        # self.multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(self.noraml_mean_weight, self.noraml_covariance_matrix)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            transfer_to=None,
            sen_hidden=None,
            bt_sen_representation=None,
            sen_id=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
            transfer_to=transfer_to,
            sen_id=sen_id,
            # project_linear=project_linear,
        )

        return encoder_outputs

    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class LongTextSTEncoderAndInter_Content_Dis_And_Attention(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.encoder = T5Stack(config, embed_tokens)
        if embed_tokens is not None:
            self.shared = embed_tokens
        # style embedding
        style_num = 3
        self.style_embedding = nn.Embedding(style_num, config.d_model)
        self.transfer_size_for_style = nn.Linear(384, 768)
        self.transfer_size_for_content = nn.Linear(384, 768)
        self.project_cycle_content = nn.Linear(768, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # attention weight
        self.attention_project = nn.Linear(768, 768)
        self.attention_project_content = nn.Linear(768, 768)
        self.attention_project_hidden = nn.Linear(768, 768)
        self.attention_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attention_project_output = nn.Linear(768, 768)
        # self.transfer_size_for_content = nn.Linear(384, 768)

        # style and content interaction
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        content_representation = sen_representation[:, 384:]
        style_representation = sen_representation[:, :384]

        sen_re_list = []
        # content_re_list = []
        for i in range(batch):
            dim_index = torch.where(posi[0] == i)
            # p_1 = posi[0][dim_index]
            single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            sen_re_list.append(single_sen)
            # content_re_list.append(single_con)

        style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        return style_re, content_representation

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden




    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                transfer_to=None,
                bt_sen_representation=None,
                sen_id=None,
                ):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if inputs_embeds is None and input_ids is not None and transfer_to is not None:
            assert sen_id != None, "sen_id is None"
            # content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids, hidden_states, sen_id)
            style_representation, content_representation = self.get_sen_representation(input_ids, hidden_states,
                                                                                   sen_id)
            style_representation = self.transfer_size_for_style(style_representation)
            content_representation = self.transfer_size_for_content(content_representation)
            trans_to_style_emb = self.style_embedding(transfer_to)
            # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
            disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
                                          trans_to_style_emb)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
            # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
            inter_hidden = self.style_content_interaction_module(disturb_hidden)

            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                style_representation=style_representation,
                content_representation=content_representation,
                sen_hidden=inter_hidden,
                # sen_hidden=new_hidden,
                # sen_mask=sen_mask,
                # batch_content=batch_content,
            )
        # elif inputs_embeds is None and input_ids is not None and transfer_to is not None and bt_sen_representation is None:
        #     assert sen_id != None, "sen_id is None"
        #     content_representation, batch_content, style_representation, posi = self.get_sen_representation(input_ids,
        #                                                                                                     hidden_states,
        #                                                                                                     sen_id)
        #     style_representation = self.transfer_size_for_style(style_representation)
        #     trans_to_style_emb = self.style_embedding(transfer_to)
        #     pred_bt_sen_representation = self.project_content_768(content_representation)
        #     disturb_hidden = self.disturb(pred_bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        #     # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), disturb_hidden), dim=1)
        #     inter_hidden = self.style_content_interaction_module(disturb_hidden)
        #     return LongTextSTEncoderAndInter_Output(
        #         last_hidden_state=encoder_outputs.last_hidden_state,
        #         # hidden_states=encoder_outputs.hidden_states,
        #         # attentions=encoder_outputs.attentions,
        #         style_representation=style_representation,
        #         content_representation=content_representation,
        #         sen_hidden=inter_hidden,
        #         # sen_mask=sen_mask,
        #         batch_content=batch_content,
        #     )

        elif inputs_embeds is not None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        elif input_ids is not None and transfer_to is None:
            return LongTextSTEncoderAndInter_Output(
                last_hidden_state=encoder_outputs.last_hidden_state,
                # hidden_states=encoder_outputs.hidden_states,
                # attentions=encoder_outputs.attentions,
                # style_representation=style_representation,
                # content_representation=content_representation,
                # sen_hidden=sen_hidden,
                # sen_mask=sen_mask,
            )
        # return LongTextSTEncoderAndInter_Output(
        #     last_hidden_state=hidden_states,
        #     past_key_values=present_key_value_states,
        #     hidden_states=all_hidden_states,
        #     attentions=all_attentions,
        #     cross_attentions=all_cross_attentions,
        # )



class LongTextST_Content_Dis_And_Attention(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = LongTextSTEncoderAndInter_Content_Dis_And_Attention(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.project_content = nn.Linear(384, 384)
        self.project_content_768 = nn.Linear(384, 768)
        # self.project_cycle_content = nn.Linear(768, 384)
        self.init_weights()

        self.noraml_mean_weight = torch.nn.Parameter(torch.zeros(config.d_model, requires_grad=True))
        self.noraml_covariance_matrix = torch.nn.Parameter(torch.ones(config.d_model, requires_grad=True))
        # self.multivariate_normal = torch.distributions.multivariate_normal.MultivariateNormal(self.noraml_mean_weight, self.noraml_covariance_matrix)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            transfer_to=None,
            sen_hidden=None,
            bt_sen_representation=None,
            sen_id=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                transfer_to=transfer_to,
                bt_sen_representation=bt_sen_representation,
                sen_id=sen_id,
            )

        # hidden_states = encoder_outputs[0]
        sen_hidden = encoder_outputs.sen_hidden

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=sen_hidden,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder_outputs = self.decoder(
        #     input_ids=decoder_input_ids,
        #     attention_mask=decoder_attention_mask,
        #     inputs_embeds=decoder_inputs_embeds,
        #     past_key_values=past_key_values,
        #     encoder_hidden_states=hidden_states,
        #     encoder_attention_mask=attention_mask,
        #     head_mask=decoder_head_mask,
        #     cross_attn_head_mask=cross_attn_head_mask,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-100)
        #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        #     # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=encoder_outputs.style_representation,
            content_representation=encoder_outputs.content_representation,
            batch_content=encoder_outputs.batch_content,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
            transfer_to=transfer_to,
            sen_id=sen_id,
            # project_linear=project_linear,
        )

        return encoder_outputs

    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        encoder_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(encoder_outputs=encoder_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class InterActionModule(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id, pad_emb):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        # style_representation, content_representation = self.get_sen_representation(input_ids, hidden_states, sen_id)
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id, pad_emb)
        content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=content_representation,
            content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


class InterActionModule_token_mean(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id, pad_emb):
        sen_index = torch.where(input_ids == sen_id)
        batch = input_ids.size(0)
        # posi = torch.where(input_ids == sen_id)
        # sen_representation = encoder_output[sen_index]
        # sen_representation = encoder_output[sen_index]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_list = []
        for i in range(batch):
            batch_sen = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                sen_content = torch.mean(encoder_output[i, m:n], dim=0, keepdim=True)
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                # batch_score.append(score_sen)
                batch_sen.append(sen_content)
                # sen_num += 1
            # single_batch_score = torch.cat((batch_score), dim=-1)
            batch_sen_tensor = torch.cat((batch_sen), dim=0)
            content_list.append(batch_sen_tensor)
        # for i in range(batch):
        #     # batch_content = []
        #     dim_index = torch.where(posi[0∂] == i)
        #     # for j in range(len(dim_index)):
        #     # p_1 = posi[0][dim_index]
        #     # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
        #     # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
        #     # batch_content.append(sen_representation[dim_index])
        #     # sen_re_list.append(single_sen)
        #     content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        # style_representation, content_representation = self.get_sen_representation(input_ids, hidden_states, sen_id)
        # content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id, pad_emb)
        trans_hidden = self.get_sen_representation(input, hidden_states, sen_id, pad_emb)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


class T5ForLongText_ST(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.mid_module = InterActionModule(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class InterActionModule_without_sty(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        sen_index = torch.where(input_ids == sen_id)
        batch = input_ids.size(0)
        style_representation = encoder_output[sen_index]
        # posi = torch.where(input_ids == sen_id)
        # sen_representation = encoder_output[sen_index]
        # sen_representation = encoder_output[sen_index]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_list = []
        style_list = []
        for i in range(batch):
            dim_index = torch.where(sen_index[0] == i)
            style_list.append(torch.mean(style_representation[dim_index], dim=0, keepdim=True))

            batch_sen = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                sen_content = torch.mean(encoder_output[i, m:n], dim=0, keepdim=True)
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                # batch_score.append(score_sen)
                batch_sen.append(sen_content)
                # sen_num += 1
            # single_batch_score = torch.cat((batch_score), dim=-1)
            batch_sen_tensor = torch.cat((batch_sen), dim=0)
            content_list.append(batch_sen_tensor)
        # for i in range(batch):
        #     # batch_content = []
        #     dim_index = torch.where(posi[0∂] == i)
        #     # for j in range(len(dim_index)):
        #     # p_1 = posi[0][dim_index]
        #     # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
        #     # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
        #     # batch_content.append(sen_representation[dim_index])
        #     # sen_re_list.append(single_sen)
        #     content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_list, batch_first=True)
        batch_style_representation = torch.cat(style_list, dim=0)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return trans_hidden, batch_style_representation

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        # style_representation = encoder_outputs[0][:, 0]
        # hidden_states = encoder_outputs[0][:, 1:, :]
        # input = input_ids[:, 1:]
        # style_representation, content_representation = self.get_sen_representation(input_ids, hidden_states, sen_id)
        # content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id, pad_emb)
        trans_hidden, style_representation = self.get_sen_representation(input_ids, encoder_outputs.last_hidden_state, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


class T5ForLongText_ST_without_sty(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)
        self.mid_module = InterActionModule_without_sty(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class InterActionModule_Sen_Sty(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        # style_representation = encoder_outputs[0][:, 0]
        # hidden_states = encoder_outputs[0][:, 1:, :]
        # input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input_ids, encoder_outputs[0], sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            # style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # no_style_sen_hidden=disturb_hidden[:,1:],
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )



class T5ForLongText_ST_Sen_Sty(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # self.sen_type_embedding = nn.Embedding(100, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        # self.encoder = T5Stack_add_sen_type(encoder_config, self.shared, self.sen_type_embedding)

        # self.mid_module = InterActionModule(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)
        self.pointer_network = Pointer_Network(encoder_config)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_sentence_types=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
        pointing_res=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # input_sentence_types=input_sentence_types,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )
            # if input_sentence_types is not None:
            pointing_res = self.pointer_network(
                    inputs_embeds=trans_output.sen_hidden[:,1:]
                    # 修改为未和style embeding进行交互的sen
                    # inputs_embeds=trans_output.content_representation
                )
            

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
            pointing_res=pointing_res,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id, input_sentence_types):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            # input_sentence_types=input_sentence_types,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None, input_sentence_types=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id, input_sentence_types)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]




# 获得所有的token
class InterActionModule_Sen_Sty_get_all_token(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        # 去掉提取sen
        # content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        # 直接使用hidden state
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), hidden_states], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=hidden_states,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )



# 获得所有token的消融
class T5ForLongText_ST_Sen_Sty_get_all_token(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_get_all_token(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]


class InterActionModule_Sen_Sty_Ablation_Sen(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            sen_re = torch.mean(sen_representation[dim_index], dim=0, keepdim=True)
            content_re_list.append(sen_re)

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )



class T5ForLongText_ST_Sen_Sty_Ablation_Sen(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_Ablation_Sen(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class InterActionModule_Sen_Sty_ProSen(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)
        self.project_sen_1 = nn.Linear(768, 768, bias=False)
        self.project_sen_2 = nn.Linear(768, 768, bias=False)
        self.project_sen_3 = nn.Linear(768, 768, bias=False)
        self.project_sen_4 = nn.Linear(768, 768, bias=False)
        self.project_sen_5 = nn.Linear(768, 768, bias=False)
        self.project_sen_6 = nn.Linear(768, 768, bias=False)
        self.project_sen_7 = nn.Linear(768, 768, bias=False)
        self.project_sen_8 = nn.Linear(768, 768, bias=False)


    def project_sen_fix_len(self, trans_hidden):
        pro_layer = [self.project_sen_1, self.project_sen_2, self.project_sen_3, self.project_sen_4, self.project_sen_5, self.project_sen_6, self.project_sen_7, self.project_sen_8]
        hidden_list = []
        for i in pro_layer:
            hidden_list.append(torch.mean(i(trans_hidden), dim=1, keepdim=True)) # b, l, n -> b, 1, n -> b, 8, n
        pro_sen = torch.cat(hidden_list, dim=1)

        return pro_sen

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        pro_hidden = self.project_sen_fix_len(trans_hidden)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, pro_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


class Project_Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs



class InterActionModule_Sen_Sty_ProSen_Att(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)
        self.n_adopter = 6
        self.key_value_proj_dim_adopter = 128
        self.inner_dim = self.n_adopter * self.key_value_proj_dim_adopter
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.key_value_proj_dim_adopter, config.d_model, bias=False)
        self.context = nn.Linear(self.key_value_proj_dim_adopter, self.key_value_proj_dim_adopter, bias=False)


    def project_sen_fix_len(self, trans_hidden):
        batch_size = trans_hidden.size(0)
        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_adopter, self.key_value_proj_dim_adopter).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer):
            """projects hidden states correctly to key/query states"""
            hidden_states = shape(proj_layer(hidden_states))

            return hidden_states

        def project_context(hidden_states):
            context = self.context(torch.mean(hidden_states, dim=-2))
            return context

        # get query states
        query_states = shape(self.q(trans_hidden))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            trans_hidden, self.k
        )
        key_context = project_context(key_states)

        # get context states
        value_states = project(
            trans_hidden, self.v
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_context.unsqueeze(-2).transpose(2, 3)
        )
        attn_weights = nn.functional.softmax(scores.float(), dim=-2).type_as(
            scores
        )
        # attn_output = torch.matmul(attn_weights, value_states.transpose(2, 3))
        attn_output = torch.matmul(value_states.transpose(2, 3), attn_weights).squeeze(-1)
        # attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)
        return attn_output

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        pro_hidden = self.project_sen_fix_len(trans_hidden)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, pro_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )




class T5ForLongText_ST_Sen_Sty_ProSen(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_ProSen_Att(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]





class T5ForLongText_ST_Sen_Sty_ProSen_Att(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_ProSen_Att(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]






##
class InterActionModule_Sen_Sty_ProSen_Att_Fix_Len(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)
        #
        self.n_adopter = 6
        # self.key_value_proj_dim_adopter = 128
        # self.inner_dim = self.n_adopter * self.key_value_proj_dim_adopter
        # self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.o = nn.Linear(self.key_value_proj_dim_adopter, config.d_model, bias=False)
        # self.context = nn.Linear(self.key_value_proj_dim_adopter, self.key_value_proj_dim_adopter, bias=False)

        # learnable hidden
        self.learnable_hidden = nn.Parameter(torch.randn(self.n_adopter, config.d_model))
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.o = nn.Linear(config.d_model, config.d_model)




    def project_sen_fix_len(self, trans_hidden):
        # batch_size = trans_hidden.size(0)

        key_states = self.k(self.learnable_hidden)
        value_states = self.v(trans_hidden)

        scores = torch.matmul(trans_hidden, key_states.transpose(0, 1))
        attn_weights = nn.functional.softmax(scores.transpose(1, 2), dim=-1)
        # a = attn_weights.detach().cpu().numpy()

        attn_output = torch.matmul(attn_weights, value_states)



        # def shape(states):
        #     """projection"""
        #     return states.view(batch_size, -1, self.n_adopter, self.key_value_proj_dim_adopter).transpose(1, 2)
        #
        # def unshape(states):
        #     """reshape"""
        #     return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        #
        # def project(hidden_states, proj_layer):
        #     """projects hidden states correctly to key/query states"""
        #     hidden_states = shape(proj_layer(hidden_states))
        #
        #     return hidden_states
        #
        # def project_context(hidden_states):
        #     context = self.context(torch.mean(hidden_states, dim=-2))
        #     return context

        # get query states
        # query_states = shape(self.q(trans_hidden))  # (batch_size, n_heads, seq_length, dim_per_head)
        #
        # # get key/value states
        # key_states = project(
        #     trans_hidden, self.k
        # )
        # key_context = project_context(key_states)
        #
        # # get context states
        # value_states = project(
        #     trans_hidden, self.v
        # )
        #
        # # compute scores
        # scores = torch.matmul(
        #     query_states, key_context.unsqueeze(-2).transpose(2, 3)
        # )
        # attn_weights = nn.functional.softmax(scores.float(), dim=-2).type_as(
        #     scores
        # )
        # # attn_output = torch.matmul(attn_weights, value_states.transpose(2, 3))
        # attn_output = torch.matmul(value_states.transpose(2, 3), attn_weights).squeeze(-1)
        # # attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)


        return attn_output

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        pro_hidden = self.project_sen_fix_len(trans_hidden)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, pro_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


class T5ForLongText_ST_Sen_Sty_ProSen_Att_Fix_Len(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.mid_module = InterActionModule_Sen_Sty_ProSen_Att_Fix_Len(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen_Att_Fix_Len_Self(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen_Att(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]




class InterActionModule_Sen_token_mean(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.style_embedding = nn.Embedding(3, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)
        #
        self.n_adopter = 6
        # self.key_value_proj_dim_adopter = 128
        # self.inner_dim = self.n_adopter * self.key_value_proj_dim_adopter
        # self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        # self.o = nn.Linear(self.key_value_proj_dim_adopter, config.d_model, bias=False)
        # self.context = nn.Linear(self.key_value_proj_dim_adopter, self.key_value_proj_dim_adopter, bias=False)

        # learnable hidden
        self.learnable_hidden = nn.Parameter(torch.randn(self.n_adopter, config.d_model))
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.o = nn.Linear(config.d_model, config.d_model)


    def project_sen_fix_len(self, trans_hidden):
        # batch_size = trans_hidden.size(0)

        key_states = self.k(self.learnable_hidden)
        value_states = self.v(trans_hidden)

        scores = torch.matmul(trans_hidden, key_states.transpose(0, 1))
        attn_weights = nn.functional.softmax(scores.transpose(1, 2), dim=-1)
        # a = attn_weights.detach().cpu().numpy()

        attn_output = torch.matmul(attn_weights, value_states)



        # def shape(states):
        #     """projection"""
        #     return states.view(batch_size, -1, self.n_adopter, self.key_value_proj_dim_adopter).transpose(1, 2)
        #
        # def unshape(states):
        #     """reshape"""
        #     return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        #
        # def project(hidden_states, proj_layer):
        #     """projects hidden states correctly to key/query states"""
        #     hidden_states = shape(proj_layer(hidden_states))
        #
        #     return hidden_states
        #
        # def project_context(hidden_states):
        #     context = self.context(torch.mean(hidden_states, dim=-2))
        #     return context

        # get query states
        # query_states = shape(self.q(trans_hidden))  # (batch_size, n_heads, seq_length, dim_per_head)
        #
        # # get key/value states
        # key_states = project(
        #     trans_hidden, self.k
        # )
        # key_context = project_context(key_states)
        #
        # # get context states
        # value_states = project(
        #     trans_hidden, self.v
        # )
        #
        # # compute scores
        # scores = torch.matmul(
        #     query_states, key_context.unsqueeze(-2).transpose(2, 3)
        # )
        # attn_weights = nn.functional.softmax(scores.float(), dim=-2).type_as(
        #     scores
        # )
        # # attn_output = torch.matmul(attn_weights, value_states.transpose(2, 3))
        # attn_output = torch.matmul(value_states.transpose(2, 3), attn_weights).squeeze(-1)
        # # attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)


        return attn_output

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(torch.mean(sen_representation[dim_index], dim=0, keepdim=True))

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # pro_hidden = self.project_sen_fix_len(trans_hidden)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        style_representation = encoder_outputs[0][:, 0]
        hidden_states = encoder_outputs[0][:, 1:, :]
        input = input_ids[:, 1:]
        content_representation, trans_hidden = self.get_sen_representation(input, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )





class T5ForLongText_ST_Sen_token_mean(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.mid_module = InterActionModule_Sen_token_mean(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen_Att_Fix_Len_Self(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen(encoder_config)
        # self.mid_module = InterActionModule_Sen_Sty_ProSen_Att(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )

        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]



class InterActionModule_Sen_Sty_En(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.style_embedding = nn.Embedding(3, config.d_model)
        self.style_embedding = nn.Embedding(2, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            content_re_list.append(sen_representation[dim_index])

        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        # style_representation = encoder_outputs[0][:, 0]
        # hidden_states = encoder_outputs[0][:, 1:, :]
        # input = input_ids[:, 1:]
        hidden_states = encoder_outputs[0]
        content_representation, trans_hidden = self.get_sen_representation(input_ids, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            # style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )



class T5ForLongText_ST_Sen_Sty_En(T5PreTrainedModel):
    
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_En(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )


        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]
        

class InterActionModule_Sen_Sty_token_mean_En(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # self.style_embedding = nn.Embedding(3, config.d_model)
        self.style_embedding = nn.Embedding(2, config.d_model)
        self.interaction_module = StyleContentInter(config, num_layers=2)

    def get_sen_representation(self, input_ids, encoder_output, sen_id):
        batch = input_ids.size(0)
        posi = torch.where(input_ids == sen_id)
        sen_representation = encoder_output[posi]
        # content_representation = sen_representation[:, 384:]
        # style_representation = sen_representation[:, :384]
        #
        # sen_re_list = []
        # all_content__list = []
        content_re_list = []
        for i in range(batch):
            # batch_content = []
            dim_index = torch.where(posi[0] == i)
            # for j in range(len(dim_index)):
            # p_1 = posi[0][dim_index]
            # single_sen = torch.mean(style_representation[dim_index], dim=0, keepdim=True)
            # single_con = torch.mean(content_representation[dim_index], dim=0, keepdim=True)
            # batch_content.append(sen_representation[dim_index])
            # sen_re_list.append(single_sen)
            # content_re_list.append(sen_representation[dim_index])
            content_re_list.append(torch.mean(sen_representation[dim_index], dim=0, keepdim=True))


        trans_hidden = pad_sequence(content_re_list, batch_first=True)
        # style_re = torch.cat(sen_re_list, dim=0)
        # content_re = torch.cat(content_re_list, dim=0)
        # style_re = self.transfer_size_for_style(torch.cat(sen_re_list, dim=0))
        # content_representation = self.transfer_size_for_content(content_representation)

        # return content_representation, content_re, style_re, posi[0]
        # return style_re, content_representation
        return sen_representation, trans_hidden

    def style_content_interaction_module(self, hidden):
        # prepare index
        # pad_emb = self.shared(torch.tensor(self.config.pad_token_id).cuda()).unsqueeze(0)
        # batch = trans_style_emb.size(0)
        # index_list = pad_sequence([torch.where(posi == i)[0] for i in range(batch)], batch_first=True, padding_value=-1)
        # # tensor pool
        # content_representation = self.transfer_size_for_content(content_representation)
        # tensor_pool = torch.cat((content_representation, pad_emb), dim=0)
        # # index gather
        # batch_sen_hidden = tensor_pool[index_list]
        # sen_ids = index_list.ne(-1).long()
        # sen_mask = torch.cat((torch.ones(batch, dtype=torch.long).to(sen_ids.device).unsqueeze(-1), sen_ids), dim=-1)
        # batch_sen_hidden_add_style = torch.cat((trans_style_emb.unsqueeze(1), batch_sen_hidden), dim=1)
        # interaction
        hidden = self.interaction_module(inputs_embeds=hidden)

        return hidden.last_hidden_state

    def disturb(self, content_representation, hidden, input_ids, sen_id, transfer_to_emb):
        sen_index = torch.where(input_ids == sen_id)
        # batch_token_index, token_index = torch.where(input_ids != sen_id)
        batch = input_ids.size(0)
        sen_num = 0
        pad_score = torch.tensor([-1e5]).to(sen_id.device)
        all_score = []
        for i in range(batch):
            batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                score_sen = torch.cat((score, pad_score), dim=0)
                batch_score.append(score_sen)
                sen_num += 1
            single_batch_score = torch.cat((batch_score), dim=-1)
            all_score.append(single_batch_score)
        score_mat = pad_sequence(all_score, batch_first=True, padding_value=-1e5)
        # score_mat_add_eos = torch.cat((score_mat, torch.ones(batch, 1).to(sen_id.device)), dim=-1)
        score_mat_add_eos = torch.cat((score_mat, pad_score.expand(batch).unsqueeze(-1)), dim=-1) / (self.config.d_model ** 0.2)
        score_mat_add_eos = F.softmax(score_mat_add_eos, dim=-1)
        # score_mat_add_eos[sen_index] = 1
        # a = score_mat_add_eos.detach().cpu().numpy()
        project_hidden = self.attention_project_hidden(hidden)
        new_hidden = torch.mul(project_hidden, score_mat_add_eos.unsqueeze(-1))
        out_hidden = self.attention_project_output(new_hidden) + hidden
        trans_hidden = self.transfer_hidden(sen_index, out_hidden, transfer_to_emb, input_ids)
        skip_hidden = self.attention_layer_norm(trans_hidden)
        # return new_hidden
        # return trans_hidden
        return skip_hidden

    def transfer_hidden(self, sen_index, hidden, transfer_emd, input_ids):
        mask = torch.ones_like(input_ids).to(hidden.device).unsqueeze(-1)
        mask[sen_index] = 0
        mask_hidden = torch.mul(hidden, mask)

        mask = mask + 1
        mask[mask == 2] = 0
        mask_transfer = torch.mul(transfer_emd.unsqueeze(1).expand_as(hidden), mask)
        transfer_hidden = mask_hidden + mask_transfer
        return transfer_hidden

    def sim_content_tokens(self, tokens, content):
        tokens_aff = self.attention_project(tokens)
        content_aff = self.attention_project_content(content)
        matmul = torch.matmul(tokens_aff, content_aff)
        # b = matmul.detach().cpu().numpy()

        # cos = F.cosine_similarity(tokens_aff, content.unsqueeze(0).expand_as(tokens))
        # attention_score_v2 = F.normalize(matmul.unsqueeze())
        # attention_score = F.sigmoid(matmul)
        # attention_score = F.softmax(matmul, dim=-1)
        # reverse_score = 1 - attention_score
        # a = matmul.detach().cpu().numpy()
        # print(a)

        return matmul

    def get_content_sen_label(self, hidden_states, input, sen_id):
        sen_index = torch.where(input == sen_id)
        batch = input.size(0)
        # sen_num = 0
        all_content = []
        for i in range(batch):
            # batch_score = []
            end = sen_index[1][sen_index[0] == i]
            start = torch.cat((torch.zeros(1, dtype=torch.int).to(end.device), end+1), dim=0)[:-1]
            for m, n in zip(start, end):
                # score = self.sim_content_tokens(hidden[i, m:n], content_representation[sen_num])
                con_i = torch.mean(hidden_states[i, m:n], dim=0, keepdim=True)
                # score_sen = torch.cat((score, torch.ones(1).to(sen_id.device)), dim=0)
                # score_sen = torch.cat((score, pad_score), dim=0)
                all_content.append(con_i)
                # sen_num += 1
        content_label = torch.cat((all_content), dim=0)

        return content_label


    def forward(
        self,
        encoder_outputs=None,
        transfer_to=None,
        sen_id=None,
        input_ids=None,
        pad_emb=None,
        ):
        # style_representation = encoder_outputs[0][:, 0]
        # hidden_states = encoder_outputs[0][:, 1:, :]
        # input = input_ids[:, 1:]
        hidden_states = encoder_outputs[0]
        content_representation, trans_hidden = self.get_sen_representation(input_ids, hidden_states, sen_id)
        # content_label = self.get_content_sen_label(hidden_states, input, sen_id)
        # style_representation = self.transfer_size_for_style(style_representation)
        # content_representation = self.transfer_size_for_content(content_representation)
        trans_to_style_emb = self.style_embedding(transfer_to)
        # disturb_hidden = self.disturb(bt_sen_representation, hidden_states, input_ids, sen_id, trans_to_style_emb)
        # disturb_hidden = self.disturb(content_representation, hidden_states, input_ids, sen_id,
        #                               trans_to_style_emb)
        # new_hidden = torch.cat((trans_to_style_emb.unsqueeze(1), hidden_states), dim=1)
        # inter_hidden = self.style_content_interaction_module(disturb_hidden)
        disturb_hidden = torch.cat([trans_to_style_emb.unsqueeze(1), trans_hidden], dim=1)
        inter_hidden = self.style_content_interaction_module(disturb_hidden)

        return LongTextSTEncoderAndInter_Output(
            last_hidden_state=encoder_outputs.last_hidden_state,
            # hidden_states=encoder_outputs.hidden_states,
            # attentions=encoder_outputs.attentions,
            # style_representation=style_representation,
            content_representation=trans_hidden,
            # content_label=content_label,
            sen_hidden=inter_hidden,
            # sen_hidden=new_hidden,
            # sen_mask=sen_mask,
            # batch_content=batch_content,
        )


        
class T5ForLongText_ST_Sen_Sty_token_mean_En(T5PreTrainedModel):
    
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # self.mid_module = InterActionModule(encoder_config)
        self.mid_module = InterActionModule_Sen_Sty_token_mean_En(encoder_config)
        # self.mid_module = InterActionModule_token_mean(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        transfer_to=None,
        sen_id=None,
        trans_output=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if trans_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            trans_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,
            )


        hidden_states = trans_output.sen_hidden

        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return StyleTransOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=trans_output.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
            style_representation=trans_output.style_representation,
            content_representation=trans_output.content_representation,
            content_label=trans_output.content_label,
            # batch_content=trans_output.batch_content,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


    def get_encoder_outputs_first(self, input_ids, transfer_to, sen_id):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        mid_output = self.mid_module(
                encoder_outputs=encoder_outputs,
                transfer_to=transfer_to,
                sen_id=sen_id,
                input_ids=input_ids,)

        return mid_output



    def inference(self, input_ids=None, decoder_start_token_id=None, top_p=None, max_length=None,
                  temperature=None, transfer_to=None, eos_id=None, return_logits=False, sen_id=None):
        batch = input_ids.size(0)
        decoder_input = torch.ones(batch).long().to(input_ids.device).unsqueeze(-1) * decoder_start_token_id
        is_done = torch.ones_like(decoder_input) * eos_id
        trans_output = self.get_encoder_outputs_first(input_ids, transfer_to, sen_id)

        for i in range(max_length):
            step_output = self(trans_output=trans_output, decoder_input_ids=decoder_input)
            next_token_logits = step_output.logits[:, -1, :] / temperature
            next_logist = top_k_top_p_filtering(logits=next_token_logits, top_p=top_p)
            probs = F.softmax(next_logist, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            # end sample add end_id
            if eos_id in decoder_input[:, -1]:
                done_sample = torch.where(decoder_input[:, -1] == eos_id)
                next_tokens[done_sample] = eos_id

            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)
            if torch.equal(next_tokens, is_done):
                break

        if return_logits:
            return step_output.logits, decoder_input[:, 1:]
        else:
            return decoder_input[:, 1:]
        