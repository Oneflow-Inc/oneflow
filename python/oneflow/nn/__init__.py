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

from oneflow.nn.graph import Graph
from oneflow.nn.module import Module
from oneflow.nn.modules.activation import (
    ELU,
    GELU,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)
from oneflow.nn.modules.adaptive_pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
)
from oneflow.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from oneflow.nn.modules.container import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from oneflow.nn.modules.conv import Conv1d, Conv2d
from oneflow.nn.modules.dataset import (
    COCOReader,
    CoinFlip,
    CropMirrorNormalize,
    OFRecordImageDecoder,
    OFRecordImageDecoderRandomCrop,
    OfrecordRawDecoder,
    OfrecordReader,
)
from oneflow.nn.modules.deconv import ConvTranspose2d
from oneflow.nn.modules.dropout import Dropout
from oneflow.nn.modules.flatten import Flatten
from oneflow.nn.modules.instancenorm import (
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
)
from oneflow.nn.modules.linear import Identity, Linear
from oneflow.nn.modules.loss import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    CTCLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    NLLLoss,
    SmoothL1Loss,
)
from oneflow.nn.modules.normalization import GroupNorm, LayerNorm
from oneflow.nn.modules.padding import (
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad2d,
    ReplicationPad2d,
)
from oneflow.nn.modules.pixelshuffle import PixelShufflev2 as PixelShuffle
from oneflow.nn.modules.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from oneflow.nn.modules.sparse import Embedding
from oneflow.nn.modules.upsampling import (
    Upsample,
    UpsamplingBilinear2d,
    UpsamplingNearest2d,
)
from oneflow.nn.modules.zeropad2d import ZeroPad2d
from oneflow.nn.parameter import Parameter
from oneflow.ops.domain_ops import (
    api_fused_self_attention_query_mul_key_and_value as fused_self_attention_query_mul_key_and_value,
)
from oneflow.ops.loss_ops import ctc_greedy_decoder
from oneflow.ops.math_ops import fused_scale_tril as fused_scale_tril
from oneflow.ops.math_ops import (
    fused_scale_tril_softmax_dropout as fused_scale_tril_softmax_dropout,
)
from oneflow.ops.math_ops import relu as relu
from oneflow.ops.math_ops import tril as tril
from oneflow.ops.nn_ops import (
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    batch_normalization,
    bias_add,
    conv1d,
    conv2d,
    conv3d,
)
from oneflow.ops.nn_ops import deconv2d as conv2d_transpose
from oneflow.ops.nn_ops import deconv2d_torch as torch_conv2d_transpose
from oneflow.ops.nn_ops import (
    distributed_sparse_softmax_cross_entropy_with_logits,
    dropout,
    elu,
    fused_bias_add_dropout,
    fused_bias_add_gelu,
)
from oneflow.ops.nn_ops import hard_sigmoid as hardsigmoid
from oneflow.ops.nn_ops import (
    hardswish,
    hardtanh,
    layer_norm,
    leaky_relu,
    logsoftmax,
    max_pool1d,
    max_pool2d,
    max_pool3d,
    mish,
    moments,
)
from oneflow.ops.nn_ops import pixel_shufflev2 as PixelShufflev2
from oneflow.ops.nn_ops import (
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
from oneflow.ops.nn_ops import tf_conv2d as compat_conv2d
