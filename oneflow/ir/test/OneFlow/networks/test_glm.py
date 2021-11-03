"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
import oneflow.nn.init as init
from  oneflow.nn import LayerNorm, Parameter
import oneflow.nn.functional as F


import math

def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    tensor_list = flow.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def unscaled_init_method(sigma):


    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):

    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False):

    world_size = 1
    init_method(weight)
    if return_master_weight:
        return weight
    return None

class ColumnParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.output_size_per_partition = divide(output_size, world_size)

        self.weight = Parameter(flow.Tensor(self.output_size_per_partition,
                                             self.input_size))
        # self.weight.model_parallel = True
        if bias:
            self.bias = Parameter(flow.Tensor(self.output_size_per_partition))

            # self.bias.model_parallel = True
            # with flow.no_grad():
            #     self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.output_size_per_partition, 0, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):
        input_parallel = input_

        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return output_parallel

class RowParallelLinear(flow.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel

        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.input_size_per_partition = divide(input_size, world_size)


        self.weight = Parameter(flow.Tensor(self.output_size,
                                             self.input_size_per_partition))
        # self.weight.model_parallel = True

        if bias:
            self.bias = Parameter(flow.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.master_weight = _initialize_affine_weight(
            self.weight, self.output_size, self.input_size,
            self.input_size_per_partition, 1, init_method,
            stride=stride, return_master_weight=keep_master_weight_for_test)

    def forward(self, input_):

        # if self.input_is_parallel:
        #     input_parallel = input_
        # else:
        #     input_parallel = scatter_to_model_parallel_region(input_)

        output_parallel = F.linear(input_, self.weight)

        # output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_parallel + self.bias
        else:
            output = output_parallel
        return output

class ParallelSelfAttention(flow.nn.Module):

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None, relative_encoding=False,
                 performer=False, attention_scale=1.0):
        super(ParallelSelfAttention, self).__init__()
        self.performer = performer

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        world_size = 1
        # world_size = get_model_parallel_world_size()

        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        self.relative_encoding = relative_encoding
        self.attention_scale = attention_scale

        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        if relative_encoding:
            self.relative = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False,
                                                 init_method=init_method)

        self.attention_dropout = flow.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = flow.nn.Dropout(output_dropout_prob)


    def _transpose_for_scores(self, tensor):

        #不支持
        # new_tensor_shape = tensor.size()[:-1] + \
        #                    (self.num_attention_heads_per_partition,
        #                     self.hidden_size_per_attention_head)
        new_tensor_shape = [*tensor.size()[:-1],self.num_attention_heads_per_partition,self.hidden_size_per_attention_head]

        tensor = tensor.reshape(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        zero_pad = flow.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = flow.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = flow.ones((x.size(0), x.size(1)))
            x = x * flow.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        query_length = hidden_states.size(1)

        #True
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = flow.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]


        query_layer = self._transpose_for_scores(mixed_query_layer)

        key_layer = self._transpose_for_scores(mixed_key_layer)

        value_layer = self._transpose_for_scores(mixed_value_layer)
        #False
        if self.relative_encoding:
            relative_layer = self.relative(position_embeddings)
            relative_layer = self._transpose_for_scores(relative_layer)
            rw_head_q = query_layer + r_w_bias.unsqueeze(1)
            ac_score = flow.matmul(rw_head_q, key_layer.transpose(-1, -2))
            rr_head_q = query_layer + r_r_bias.unsqueeze(1)
            bd_score = flow.matmul(rr_head_q, relative_layer.transpose(-1, -2))
            bd_score = self._rel_shift(bd_score)

            attention_scores = ac_score + bd_score
            attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        else:
            if self.attention_scale > 1.0:
                attention_scores = flow.matmul(query_layer / math.sqrt(self.attention_scale),
                                            key_layer.transpose(-1, -2) / math.sqrt(
                                                self.hidden_size_per_attention_head * self.attention_scale))
            else:
                attention_scores = flow.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(
                    self.hidden_size_per_attention_head))


        attention_scores = flow.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale

        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)

        attention_probs = flow.nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attention_dropout(attention_probs)

        context_layer = flow.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # new_context_layer_shape = context_layer.size()[:-2] + \
        #                           (self.hidden_size_per_partition,)
        new_context_layer_shape = [*context_layer.size()[:-2],self.hidden_size_per_partition]
        context_layer = context_layer.reshape(*new_context_layer_shape)

        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


if __name__ == "__main__":
    hidden_size = 512
    num_attention_heads = 8
    attention_dropout_prob = 0.1
    output_dropout_prob = 0.1
    init_method = unscaled_init_method(0.2)
    output_layer_init_method = scaled_init_method(0.2, 12)
    relative_encoding = False
    performer = False
    attention_scale = 1.0

    class AttenModel(flow.nn.Module):
        def __init__(self, ):
            super(AttenModel, self).__init__()
            self.atten = flow.nn.ModuleList(
                            [ParallelSelfAttention(
                                hidden_size,
                                num_attention_heads,
                                attention_dropout_prob,
                                output_dropout_prob,
                                init_method,
                                output_layer_init_method=output_layer_init_method,
                                relative_encoding=relative_encoding,
                                performer=performer,
                                attention_scale=attention_scale)
                                for _ in range(12)
                            ]
                        )

        def forward(self, x, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem):
            for layer in self.atten:
                x = layer(x, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
            return x

    layernorm_output = flow.randn(4, 332, 512)
    ltor_mask = flow.randn(4, 1, 332, 332)
    position_embeddings = None
    r_w_bias = None
    r_r_bias = None
    mem = None
    layernorm_output = layernorm_output.to("cuda")
    ltor_mask = ltor_mask.to("cuda")
    # ============= comment code =================
    model = AttenModel()
    model.train()
    model = model.to("cuda")
    optimizer = flow.optim.Adam(
                        model.parameters(),
                        lr=0.01,
                        )
    # ============= comment end =================
    for i in range(10000000):
        print("training")
        # ============= comment code =================
        attention_output = model(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        optimizer.zero_grad()
        loss = attention_output.sum()
        loss.backward()
        optimizer.step()
        # ============= comment end =================
