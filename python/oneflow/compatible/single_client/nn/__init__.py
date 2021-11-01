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

from oneflow.compatible.single_client.nn.module import Module
from oneflow.compatible.single_client.nn.parameter import Parameter
from oneflow.compatible.single_client.ops.domain_ops import (
    api_fused_self_attention_query_mul_key_and_value as fused_self_attention_query_mul_key_and_value,
)
from oneflow.compatible.single_client.ops.loss_ops import ctc_greedy_decoder
from oneflow.compatible.single_client.ops.math_ops import (
    fused_scale_tril as fused_scale_tril,
)
from oneflow.compatible.single_client.ops.math_ops import (
    fused_scale_tril_softmax_dropout as fused_scale_tril_softmax_dropout,
)
from oneflow.compatible.single_client.ops.math_ops import relu as relu
from oneflow.compatible.single_client.ops.math_ops import tril as tril
from oneflow.compatible.single_client.ops.nn_ops import (
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    batch_normalization,
)
from oneflow.compatible.single_client.ops.nn_ops import bce_loss as BCELoss
from oneflow.compatible.single_client.ops.nn_ops import (
    bce_with_logits_loss as BCEWithLogitsLoss,
)
from oneflow.compatible.single_client.ops.nn_ops import bias_add, conv1d, conv2d, conv3d
from oneflow.compatible.single_client.ops.nn_ops import deconv2d as conv2d_transpose
from oneflow.compatible.single_client.ops.nn_ops import (
    deconv2d_torch as torch_conv2d_transpose,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    distributed_sparse_softmax_cross_entropy_with_logits,
    dropout,
    elu,
    fused_bias_add_dropout,
    fused_bias_add_gelu,
)
from oneflow.compatible.single_client.ops.nn_ops import group_normalization as GroupNorm
from oneflow.compatible.single_client.ops.nn_ops import hard_sigmoid as hardsigmoid
from oneflow.compatible.single_client.ops.nn_ops import hardswish, hardtanh
from oneflow.compatible.single_client.ops.nn_ops import (
    instance_normalization1d as InstanceNorm1d,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    instance_normalization2d as InstanceNorm2d,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    instance_normalization3d as InstanceNorm3d,
)
from oneflow.compatible.single_client.ops.nn_ops import kldivloss as KLDivLoss
from oneflow.compatible.single_client.ops.nn_ops import l1_loss as L1Loss
from oneflow.compatible.single_client.ops.nn_ops import (
    layer_norm,
    leaky_relu,
    logsoftmax,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    margin_ranking_loss as MarginRankingLoss,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    max_pool1d,
    max_pool2d,
    max_pool3d,
    mish,
    moments,
)
from oneflow.compatible.single_client.ops.nn_ops import mse_loss as MSELoss
from oneflow.compatible.single_client.ops.nn_ops import pixel_shuffle as PixelShuffle
from oneflow.compatible.single_client.ops.nn_ops import (
    pixel_shufflev2 as PixelShufflev2,
)
from oneflow.compatible.single_client.ops.nn_ops import (
    random_mask_like,
    relu6,
    sigmoid_cross_entropy_with_logits,
    softmax,
    softmax_cross_entropy_with_logits,
    softmax_grad,
    sparse_cross_entropy,
    sparse_softmax_cross_entropy_with_logits,
    swish,
)
from oneflow.compatible.single_client.ops.nn_ops import tf_conv2d as compat_conv2d
from oneflow.compatible.single_client.ops.nn_ops import (
    triplet_margin_loss as TripletMarginLoss,
)
