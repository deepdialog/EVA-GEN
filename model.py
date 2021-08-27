
from typing import Callable, Optional, List
import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import jieba

from configuration_enc_dec import EncDecConfig


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False, input_is_parallel=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size

        self.weight = Parameter(torch.Tensor(self.output_size,
                                             self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        if self.bias is not None:
            output = F.linear(input_, self.weight, self.bias)
        else:
            output = F.linear(input_, self.weight)
        return output

RowParallelLinear = ColumnParallelLinear


class TorchAttention(nn.Module):
    def __init__(
        self,
        config: EncDecConfig, 
        init_method = None,
        is_decoder: bool = False,
        is_cross_attn: bool = False,
        output_layer_init_method = None,
        has_relative_attention_bias: bool = False):

        super(TorchAttention, self).__init__()

        self.is_decoder = is_decoder
        self.is_cross_attn = is_cross_attn
        self.output_attention = config.output_attention

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

        
        d_attn_out = config.d_kv * config.num_heads # h
        
        # Per attention head and per partition values.
        # world_size = get_model_parallel_world_size() # p
        self.hidden_size_per_partition = d_attn_out  # divide(d_attn_out, world_size) # h_p
        self.hidden_size_per_attention_head = config.d_kv # h_i
        self.num_attention_heads_per_partition = config.num_heads  # divide(config.num_heads, world_size) # n_p

        # Strided linear layer.
        if is_cross_attn:
            self.project_q = ColumnParallelLinear(config.d_model, d_attn_out,
                                                  stride=1, # NOTE: modify stride
                                                  bias=False,
                                                  gather_output=False)
            self.project_kv = ColumnParallelLinear(config.d_model, 2 * d_attn_out,
                                                   stride=2,  # NOTE: modify stride
                                                   bias=False,
                                                   gather_output=False)
        else:
            self.project = ColumnParallelLinear(config.d_model, 3 * d_attn_out,
                                                        stride=3,
                                                        bias=False,
                                                        gather_output=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.num_attention_heads_per_partition)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(config.dropout_rate)

        # Output.
        self.dense = RowParallelLinear(d_attn_out,
                                       config.d_model,
                                       input_is_parallel=True,
                                       bias=False)
        self.output_dropout = nn.Dropout(config.dropout_rate)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, h_p=n_p*h_i] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head) # [b, s, n_p, h_i]
        tensor = tensor.view(*new_tensor_shape)
        # tensor: [b, n_p, s, h_i]
        return tensor.permute(0, 2, 1, 3)

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
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        query_length=None,
        past_key_value=None,):
        
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        # hidden_states: [b, s, d_model]
        if key_value_states is not None:
            assert self.is_cross_attn is True
            # mixed_query_layer: [b, s, h_p]
            mixed_query_layer = self.project_q(hidden_states)
            # mixed_key_value_layer: [b, s, 2 * h_p]
            mixed_key_value_layer = self.project_kv(key_value_states)
            (mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_key_value_layer, 2)
        else:
            assert self.is_cross_attn is False
            # hidden_states: [b, s, h]
            mixed_x_layer = self.project(hidden_states)
            # mixed_x_layer: [b, s, 3 * h_p]
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            # mixed_***_layer: [b, s, h_p]

        # ***_layer [b, n_p, s, h_i]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if past_key_value is not None and not self.is_cross_attn:
            assert self.is_decoder is True
            # decoder
            # ***_layer: [b, n_p, 1, h_i]
            past_key_layer, past_value_layer = past_key_value
            # past_***_layer: [b, n_p, s-1, h_i]
            key_layer = torch.cat([past_key_layer, key_layer], dim=2)
            value_layer = torch.cat([past_value_layer, value_layer], dim=2)
            # ***_layer: [b, n_p, s_k, h_i]

        # Raw attention scores. [b, n_p, s_q, s_k] compute every head alone
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        # NOTE: We follow the implementation of Transformers to remove the scale of attention+acores
        # attention_scores = attention_scores / math.sqrt(
        #     self.hidden_size_per_attention_head)
        
        # relative positional bias
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads_per_partition, real_seq_length, key_length),
                    device=attention_scores.device, dtype=attention_scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
            
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

        no_pos_bias_attn_probs = nn.Softmax(dim=-1)(attention_scores)
        # Apply the attention mask [b, 1, s_q, s_k] and relative position_bias
        # NOTE: 10000 can't be larger otherwise may cause fp16 overflow (max in fp16 = 65504)
        # if attention_mask is not None:
        if attention_scores.dtype  == torch.float16:
            attention_scores = torch.mul(
                attention_scores, attention_mask
            ) + (-10000.0 * (1.0 - attention_mask) + position_bias)
        else:
            attention_scores = torch.mul(
                attention_scores, attention_mask
            ) + (-100000000.0 * (1.0 - attention_mask) + position_bias)

        # attention_scores = torch.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)

        # Attention probabilities. [b, n_p, s_q, s_k]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with get_cuda_rng_tracker().fork():
        #     attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: [b, n_p, s, h_i]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [b, s, n_p, h_i]
        # if self.do_dim_trick:
        #     head_mask = self.head_mask.view(1, 1, self.head_mask.size(0), 1).expand_as(context_layer)
        #     context_layer = context_layer * head_mask

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: [b, s, h_p]

        attn_output = self.dense(context_layer)
        # attn_output: [b, s, d_model]
        attn_output = self.output_dropout(attn_output)

        present_key_value_state = torch.stack((key_layer, value_layer), dim=0) if self.is_decoder else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if self.output_attention:
            outputs += (no_pos_bias_attn_probs,)
        else:
            outputs += (None,)

        return outputs  # attn_output, present_key_value_state, position_bias, attention_probs
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class TorchSelfAttention(nn.Module):
    def __init__(
        self,
        config: EncDecConfig, 
        init_method: Callable=None,
        is_decoder: bool = False,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False):
        
        super(TorchSelfAttention, self).__init__()
        self.self_attn = TorchAttention(
            config, 
            init_method,
            is_decoder=is_decoder,
            is_cross_attn=False,
            output_layer_init_method=output_layer_init_method, 
            has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, (attention_probs)
    
    
class TorchCrossAttention(nn.Module):
    def __init__(
        self,
        config: EncDecConfig,
        init_method: Callable=None,
        is_decoder: bool = True,
        output_layer_init_method: Optional[Callable] = None):
        
        super(TorchCrossAttention, self).__init__()

        self.cross_attn = TorchAttention(
            config,
            init_method,
            is_decoder=is_decoder,
            is_cross_attn=True,
            output_layer_init_method=output_layer_init_method,
            has_relative_attention_bias=False)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        query_length=None,
        past_key_value=None):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.cross_attn(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            query_length=query_length,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, (attention_probs)
    
    
class TorchDenseReluDense(nn.Module):
    def __init__(self,
                 config: EncDecConfig,
                 init_method: Callable=None,
                 output_layer_init_method: Optional[Callable] = None):
        super(TorchDenseReluDense, self).__init__()
        self.wi_0 = ColumnParallelLinear(
            config.d_model, config.d_ff,
            gather_output=False,
            bias=False,
            init_method=init_method)
        self.wi_1 = ColumnParallelLinear(
            config.d_model, config.d_ff,
            gather_output=False,
            bias=False,
            init_method=init_method)
        self.wo = RowParallelLinear(
            config.d_ff,
            config.d_model,
            bias=False,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = nn.Dropout(config.dropout_rate)

        # self.do_dim_trick = config.do_dim_trick
        # if torch.distributed.get_rank() % 5 == 4:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13104 + [0.0] * 4), requires_grad=False)
        # else:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13108), requires_grad=False)

    def forward(self, hidden_states):
        # hidden_states: [b, s, hp]
        hidden_gelu = gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        # hidden_states: [b, s, d_ff_p]
        # if self.do_dim_trick:
        #     ff_mask = self.ff_mask.view(1, 1, self.ff_mask.size(0))
        #     hidden_states = ff_mask * hidden_states

        # hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        # hidden_states: [b, s, hp]
        return hidden_states
    

class TorchFF(nn.Module):
    def __init__(
        self,
        config: EncDecConfig,
        init_method: Callable=None,
        output_layer_init_method: Callable = None):
        super(TorchFF, self).__init__()

        self.dense_relu_dense = TorchDenseReluDense(config, init_method, output_layer_init_method)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # hidden_states [b, s, d_model]
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.dense_relu_dense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class TorchBlock(nn.Module):
    def __init__(
        self, 
        config: EncDecConfig,
        init_method: Callable = None,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False, 
        is_decoder: bool = False):
        super(TorchBlock, self).__init__()

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.is_decoder = is_decoder

        self.self_attn = TorchSelfAttention(
            config,
            init_method,
            is_decoder=is_decoder,
            output_layer_init_method=output_layer_init_method, 
            has_relative_attention_bias=has_relative_attention_bias)

        if is_decoder:
            self.cross_attn = TorchCrossAttention(
                config,
                init_method,
                is_decoder=is_decoder,
                output_layer_init_method=output_layer_init_method)

        self.ff = TorchFF(
            config,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        enc_hidden_states=None,
        cross_attention_mask=None,
        enc_dec_position_bias=None,
        past_key_value=None,):

        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[0]
            cross_attn_past_key_value = past_key_value[1]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states, self_attn_present_key_value = self_attn_outputs[:2]
        position_bias = (self_attn_outputs[2],)
        attention_probs = (self_attn_outputs[3],)
        present_key_value = (self_attn_present_key_value,)

        # cross attn
        if self.is_decoder:
            if self_attn_present_key_value is not None:
                query_length = self_attn_present_key_value[0].shape[2]
            else:
                query_length = None

            cross_attn_outputs = self.cross_attn(
                hidden_states,
                key_value_states=enc_hidden_states,
                attention_mask=cross_attention_mask,
                position_bias=enc_dec_position_bias,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
            )

            hidden_states, cross_attn_present_key_value = cross_attn_outputs[:2]
            present_key_value += (cross_attn_present_key_value,)
            # Keep cross-attention outputs and relative position weights
            position_bias = position_bias + (cross_attn_outputs[2],)
            attention_probs = attention_probs + (cross_attn_outputs[3],)

        hidden_states = self.ff(hidden_states)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value,) + position_bias + attention_probs

        # (for encoder) hidden_states, present_key_value_states, self-attention position bias, attention_probs
        # (for decoder) hidden_states, present_key_value_states, self-attention position bias, cross-attention position bias, self_attention_probs, cross_attention_probs
        return outputs


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.

#         self.vocab_start_index, self.vocab_end_index = \
#             VocabUtility.vocab_range_from_global_vocab_size(
#                 self.num_embeddings, get_model_parallel_rank(),
#                 get_model_parallel_world_size())
        self.vocab_start_index, self.vocab_end_index = 0, num_embeddings
        self.num_embeddings_per_partition = self.vocab_end_index - \
                                            self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition,
                                             self.embedding_dim))
        self.weight.model_parallel = True
        # And initialize.
#         _initialize_affine_weight(
#             self.weight, self.num_embeddings, self.embedding_dim,
#             self.num_embeddings_per_partition, 0, init_method)

    def forward(self, input_):
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        # output = reduce_from_model_parallel_region(output_parallel)
        output = output_parallel
        return output



class TorchTransformer(nn.Module):
    def __init__(self, config: EncDecConfig, word_embeds: VocabParallelEmbedding, data_hack=None, prompt_config=None, is_decoder=False, checkpoint_activations=False, checkpoint_num_layers=1):
        super(TorchTransformer, self).__init__()
        
        self.word_embeds = word_embeds
        self.config = config
        self.prompt_config = prompt_config
        if self.prompt_config is not None:
            self.prompt_embeds = nn.Embedding(prompt_config["prompt_len"], config.d_model)
        # self.position_embeds = nn.Embedding(config.max_position_embeddings, config.d_model)
        # init_method_normal(std=config.init_method_std)(self.position_embeds.weight)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.final_layernorm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.is_decoder = is_decoder
        self.data_hack = data_hack
        # if self.data_hack == "chid":
        #     if not self.is_decoder:
        #         print("loading /mnt/sfs_turbo/data/CLUE/chid/id_2_vec.pkl")
        #         with open("/mnt/sfs_turbo/data/CLUE/chid/id_2_vec.pkl", "rb") as f:
        #             t = pickle.load(f)
        #         t = torch.tensor(t)
        #         self.idiom_fc = nn.Linear(t.size(1), config.d_model)
        #         self.add_embeds = nn.Embedding.from_pretrained(t, freeze=False)

        output_layer_init_method = None

        self.blocks = nn.ModuleList(
            [TorchBlock(
                config,
                None,
                has_relative_attention_bias=bool(i == 0),
                output_layer_init_method=output_layer_init_method,
                is_decoder=is_decoder) for i in range(config.num_layers)]
        )

    def init_prompt_embeds(self):
        if self.prompt_config is not None:
            prompt_weights = self.word_embeds(self.prompt_config["init_ids"]).detach()
            # self.prompt_embeds = nn.Embedding(self.prompt_config["prompt_len"], self.config.d_model).from_pretrained(prompt_weights, freeze=False)
            self.prompt_embeds.weight.data = prompt_weights

    def get_input_embeds(self, input_ids):
        if self.prompt_config is None and self.data_hack is None:
            return self.word_embeds(input_ids)

        p_embeds = None
        if self.prompt_config is not None:
            prompt_mask = (input_ids < 0).long()
            prompt_ids = (-(input_ids * prompt_mask)) - prompt_mask
            p_embeds = self.prompt_embeds(prompt_ids) * prompt_mask.float().unsqueeze(-1)

        # a_embeds = None
        # if self.data_hack == "chid":
        #     if not self.is_decoder:
        #         idiom_mask = (input_ids >= self.config.vocab_size).long()
        #         idiom_ids = (input_ids * idiom_mask) - self.config.vocab_size * idiom_mask
        #         a_embeds = self.idiom_fc(self.add_embeds(idiom_ids)) * idiom_mask.float().unsqueeze(-1)

        word_mask = (0 <= input_ids).long()
        word_ids = word_mask * input_ids
        w_embeds = self.word_embeds(word_ids) * word_mask.float().unsqueeze(-1)

        if p_embeds is not None:
            w_embeds = w_embeds + p_embeds
        # if a_embeds is not None:
        #     w_embeds = w_embeds + a_embeds

        return w_embeds # bs * seq_len * hidden

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,):
        
        inputs_embeds = self.get_input_embeds(input_ids)

        # remove abstract position ids
        # pos_embeds = self.position_embeds(position_ids)
        # inputs_embeds = inputs_embeds + pos_embeds

        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        enc_dec_position_bias = None
        present_key_value_states = []

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        # NOTE: check implementation: checkpoint_activations

        all_self_attention_probs = []
        all_cross_attention_probs = []

        def custom(start, end):
            def custom_forward(*inputs):                
                layer_modules_ = self.blocks[start:end]
                past_key_values_ = past_key_values[start:end]
                self_attn_present_key_values_ = []
                cross_attn_present_key_values_ = []
                position_bias_, enc_dec_position_bias_ = None, None

                hidden_states_ = inputs[0]
                if len(inputs) > 2:
                    position_bias_ = inputs[1]
                if len(inputs) > 3:
                    enc_dec_position_bias_ = inputs[2]
                
                if enc_hidden_states is not None:
                    enc_hidden_states_ = inputs[-1]
                else:
                    enc_hidden_states_ = None

                for layer_, past_key_value_ in zip(layer_modules_, past_key_values_):
                    layer_outputs_ = layer_(hidden_states_,
                                            attention_mask,
                                            position_bias_,
                                            enc_hidden_states_,
                                            cross_attention_mask,
                                            enc_dec_position_bias_,
                                            past_key_value=past_key_value_)
                    
                    hidden_states_, present_key_value_ = layer_outputs_[:2]
                    if self.is_decoder:
                        self_attn_present_key_values_.append(present_key_value_[0])
                        cross_attn_present_key_values_.append(present_key_value_[1])
                        all_self_attention_probs.append(layer_outputs_[-2])
                        all_cross_attention_probs.append(layer_outputs_[-1])
                    else:
                        self_attn_present_key_values_.append(present_key_value_[0])
                        all_self_attention_probs.append(layer_outputs_[-1])

                    position_bias_ = layer_outputs_[2]
                    if self.is_decoder and enc_hidden_states is not None:
                        enc_dec_position_bias_ = layer_outputs_[3]
                
                outputs_ = (hidden_states_,)
                if position_bias_ is not None:
                    outputs_ += (position_bias_,)
                if enc_dec_position_bias_ is not None:
                    outputs_ += (enc_dec_position_bias_,)
                if self.is_decoder:
                    self_attn_present_key_values_ = torch.stack(self_attn_present_key_values_, dim=0)
                    cross_attn_present_key_values_ = torch.stack(cross_attn_present_key_values_, dim=0)
                    outputs_ += (self_attn_present_key_values_, cross_attn_present_key_values_,)
                return outputs_
            
            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.blocks)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                arg_list = (hidden_states,)
                if position_bias is not None:
                    arg_list += (position_bias,)
                if enc_dec_position_bias is not None:
                    arg_list += (enc_dec_position_bias,)
                
                if enc_hidden_states is not None:
                    arg_list += (enc_hidden_states,)
                    tmp_outputs = checkpoint(custom(l, l+chunk_length), *arg_list)
                else:
                    arg_list += (attention_mask,)
                    tmp_outputs = checkpoint(custom(l, l+chunk_length), *arg_list)
                
                hidden_states = tmp_outputs[0]
                if self.is_decoder:
                    if len(tmp_outputs) > 3:
                        position_bias = tmp_outputs[1]
                    if len(tmp_outputs) > 4:
                        enc_dec_position_bias = tmp_outputs[2]
                    present_key_value_states.extend([(s, c) for s, c in zip(tmp_outputs[-2], tmp_outputs[-1])])
                else:
                    if len(tmp_outputs) > 1:
                        position_bias = tmp_outputs[1]
                    if len(tmp_outputs) > 2:
                        enc_dec_position_bias = tmp_outputs[2]
                    present_key_value_states.extend([None] * chunk_length)            
                
                l += chunk_length
        else:
            for i, (layer_module, past_key_value) in enumerate(zip(self.blocks, past_key_values)):

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    enc_hidden_states=enc_hidden_states,
                    cross_attention_mask=cross_attention_mask,
                    enc_dec_position_bias=enc_dec_position_bias,
                    past_key_value=past_key_value
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, self-attention position bias, cross-attention position bias, attention_probs
                hidden_states, present_key_value_state = layer_outputs[:2]
                if self.is_decoder:
                    all_self_attention_probs.append(layer_outputs[-2])
                    all_cross_attention_probs.append(layer_outputs[-1])
                else:
                    all_self_attention_probs.append(layer_outputs[-1])

                position_bias = layer_outputs[2]
                if self.is_decoder and enc_hidden_states is not None:
                    enc_dec_position_bias = layer_outputs[3]
                
                present_key_value_states.append(present_key_value_state)
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                # position_bias = layer_outputs[2]

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        outputs = {
            "last_hidden_state": hidden_states,
            "past_key_values": present_key_value_states,
            "hidden_states": None,
            "attentions": all_self_attention_probs,
            "cross_attentions": all_cross_attention_probs
        }

        return outputs


class TorchEncDecModel(nn.Module):
    
    def __init__(
        self,
        config: EncDecConfig,
        parallel_output=True,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        prompt_config=None,
        data_hack=None):
        
        super(TorchEncDecModel, self).__init__()
        if config.vocab_size is None:
            raise RuntimeError("Should set vocab size")
        self.enc_config = copy.deepcopy(config)
        self.dec_config = copy.deepcopy(config)

        self.parallel_output = parallel_output

        init_method = None  # init_method_normal(std=config.init_method_std) # NOTE: good?

        self.word_embeds = VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.prompt_config = prompt_config

        self.lm_head = VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.encoder = TorchTransformer(
            self.enc_config, word_embeds=self.word_embeds,
            is_decoder=False, data_hack=data_hack,
            prompt_config=prompt_config["enc"] if prompt_config is not None else None,
            checkpoint_activations=checkpoint_activations,
            checkpoint_num_layers=checkpoint_num_layers)

        self.decoder = TorchTransformer(
            self.dec_config, word_embeds=self.word_embeds,
            is_decoder=True, data_hack=data_hack,
            prompt_config=None if prompt_config is not None else None,                                   
            checkpoint_activations=checkpoint_activations,
            checkpoint_num_layers=checkpoint_num_layers)

    def init_prompt_embeds(self):
        self.encoder.init_prompt_embeds()
        self.decoder.init_prompt_embeds()

    def forward(
        self, 
        enc_input_ids=None,
        enc_position_ids=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,
        only_encoder=False):

        provided_hidden = (enc_hidden_states is not None)

        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
            )

            enc_hidden_states = enc_outputs["last_hidden_state"]

        if only_encoder:
            outputs = {
                "encoder_last_hidden_state": enc_hidden_states,
                "encoder_hidden_states": enc_outputs["hidden_states"],
                "encoder_attentions": enc_outputs["attentions"],
            }

            return outputs

        dec_outputs = self.decoder(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states,
            past_key_values=past_key_values,
        )

        last_hidden_state_parallel = dec_outputs["last_hidden_state"]
        logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)

        lm_logits = logits_parallel

        outputs = {
            "lm_logits": lm_logits,
            "last_hidden_state": dec_outputs["last_hidden_state"],
            "past_key_values": dec_outputs["past_key_values"],
            "encoder_last_hidden_state": enc_hidden_states,
            "encoder_attentions": enc_outputs["attentions"] if not provided_hidden else None,
            "decoder_self_attentions": dec_outputs["attentions"],
            "decoder_cross_attentions": dec_outputs["cross_attentions"]
        }

        return outputs


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for logit in logits:
            sorted_logits, sorted_indices = torch.sort(logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits


def test():
    # l = ColumnParallelLinear(10, 20).half()
    # t = torch.rand((5, 10), dtype=torch.float16)
    # assert l(t).shape == torch.Size([5, 20])
    # # 包括bias和weight，所以是2
    # assert 2 == len(list(l.parameters()))

    # l = RowParallelLinear(10, 20, bias=False)
    # t = torch.rand((5, 10))
    # assert l(t).shape == torch.Size([5, 20])
    # # 没有bias，所以是1
    # assert 1 == len(list(l.parameters()))

    # # encoder和decoder的第0个block，has_relative_attention_bias = True
    # l = TorchAttention(config, has_relative_attention_bias=True)
    # for x in l.parameters():
    #     print(x.shape)

    # l = TorchAttention(config)
    # for x in l.parameters():
    #     print(x.shape)

    """
    encoder.blocks.0.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight torch.Size([32, 16])
    encoder.blocks.0.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    encoder.blocks.0.self_attn.layer_norm.weight torch.Size([4096])
    """
    # l = TorchSelfAttention(config, has_relative_attention_bias=True)
    # for x in l.parameters():
    #     print(x.shape)

    """
    encoder.blocks.1.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    encoder.blocks.1.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    encoder.blocks.1.self_attn.layer_norm.weight torch.Size([4096])
    """
    # l = TorchSelfAttention(config)
    # for x in l.parameters():
    #     print(x.shape)
    """
    decoder.blocks.0.cross_attn.cross_attn.project_q.weight torch.Size([1024, 4096])
    decoder.blocks.0.cross_attn.cross_attn.project_kv.weight torch.Size([2048, 4096])
    decoder.blocks.0.cross_attn.cross_attn.dense.weight torch.Size([4096, 1024])
    decoder.blocks.0.cross_attn.layer_norm.weight torch.Size([4096])
    """
    # l = TorchCrossAttention(config)
    # for x in l.parameters():
    #     print(x.shape)

    # l = TorchDenseReluDense(config)
    # for x in l.parameters():
    #     print(x.shape)

    """
    encoder.blocks.0.ff.dense_relu_dense.wi_0.weight torch.Size([2560, 4096])
    encoder.blocks.0.ff.dense_relu_dense.wi_1.weight torch.Size([2560, 4096])
    encoder.blocks.0.ff.dense_relu_dense.wo.weight torch.Size([4096, 2560])
    encoder.blocks.0.ff.layer_norm.weight torch.Size([4096])
    """
    # l = TorchFF(config)
    # for x in l.parameters():
    #     print(x.shape)

    """
    encoder.blocks.0.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight torch.Size([32, 16])
    encoder.blocks.0.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    encoder.blocks.0.self_attn.layer_norm.weight torch.Size([4096])
    encoder.blocks.0.ff.dense_relu_dense.wi_0.weight torch.Size([2560, 4096])
    encoder.blocks.0.ff.dense_relu_dense.wi_1.weight torch.Size([2560, 4096])
    encoder.blocks.0.ff.dense_relu_dense.wo.weight torch.Size([4096, 2560])
    encoder.blocks.0.ff.layer_norm.weight torch.Size([4096])
    """
    # l = TorchBlock(config, has_relative_attention_bias=True)
    # for x in l.parameters():
    #     print(x.shape)

    """
    encoder.blocks.1.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    encoder.blocks.1.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    encoder.blocks.1.self_attn.layer_norm.weight torch.Size([4096])
    encoder.blocks.1.ff.dense_relu_dense.wi_0.weight torch.Size([2560, 4096])
    encoder.blocks.1.ff.dense_relu_dense.wi_1.weight torch.Size([2560, 4096])
    encoder.blocks.1.ff.dense_relu_dense.wo.weight torch.Size([4096, 2560])
    encoder.blocks.1.ff.layer_norm.weight torch.Size([4096])
    """
    # l = TorchBlock(config, has_relative_attention_bias=False)
    # for x in l.parameters():
    #     print(x.shape)

    """
    decoder.blocks.0.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight torch.Size([32, 16])
    decoder.blocks.0.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    decoder.blocks.0.self_attn.layer_norm.weight torch.Size([4096])
    decoder.blocks.0.cross_attn.cross_attn.project_q.weight torch.Size([1024, 4096])
    decoder.blocks.0.cross_attn.cross_attn.project_kv.weight torch.Size([2048, 4096])
    decoder.blocks.0.cross_attn.cross_attn.dense.weight torch.Size([4096, 1024])
    decoder.blocks.0.cross_attn.layer_norm.weight torch.Size([4096])
    decoder.blocks.0.ff.dense_relu_dense.wi_0.weight torch.Size([2560, 4096])
    decoder.blocks.0.ff.dense_relu_dense.wi_1.weight torch.Size([2560, 4096])
    decoder.blocks.0.ff.dense_relu_dense.wo.weight torch.Size([4096, 2560])
    decoder.blocks.0.ff.layer_norm.weight torch.Size([4096])
    """
    # l = TorchBlock(config, has_relative_attention_bias=True, is_decoder=True)
    # for x in l.parameters():
    #     print(x.shape)

    """
    decoder.blocks.1.self_attn.self_attn.project.weight torch.Size([3072, 4096])
    decoder.blocks.1.self_attn.self_attn.dense.weight torch.Size([4096, 1024])
    decoder.blocks.1.self_attn.layer_norm.weight torch.Size([4096])
    decoder.blocks.1.cross_attn.cross_attn.project_q.weight torch.Size([1024, 4096])
    decoder.blocks.1.cross_attn.cross_attn.project_kv.weight torch.Size([2048, 4096])
    decoder.blocks.1.cross_attn.cross_attn.dense.weight torch.Size([4096, 1024])
    decoder.blocks.1.cross_attn.layer_norm.weight torch.Size([4096])
    decoder.blocks.1.ff.dense_relu_dense.wi_0.weight torch.Size([2560, 4096])
    decoder.blocks.1.ff.dense_relu_dense.wi_1.weight torch.Size([2560, 4096])
    decoder.blocks.1.ff.dense_relu_dense.wo.weight torch.Size([4096, 2560])
    decoder.blocks.1.ff.layer_norm.weight torch.Size([4096])
    """
    # l = TorchBlock(config, has_relative_attention_bias=False, is_decoder=True)
    # for x in l.parameters():
    #     print(x.shape)
