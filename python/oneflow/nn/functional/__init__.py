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
from oneflow.nn.modules.interpolate import interpolate
from oneflow.nn.modules.affine_grid import affine_grid
from oneflow.nn.modules.grid_sample import grid_sample
from oneflow.nn.modules.sparse_softmax_cross_entropy import sparse_softmax_cross_entropy
from oneflow._C import conv1d
from oneflow._C import conv2d
from oneflow._C import conv3d
from oneflow._C import deconv1d as conv_transpose1d
from oneflow._C import deconv2d as conv_transpose2d
from oneflow._C import deconv3d as conv_transpose3d
from oneflow._C import avg_pool1d
from oneflow._C import avg_pool2d
from oneflow._C import avg_pool3d
from .functional_maxpool import max_pool1d
from .functional_maxpool import max_pool2d
from .functional_maxpool import max_pool3d
from oneflow._C import adaptive_avg_pool1d
from oneflow._C import adaptive_avg_pool2d
from oneflow._C import adaptive_avg_pool3d
from oneflow._C import cosine_similarity
from oneflow._C import relu
from oneflow._C import hardtanh
from oneflow._C import hardsigmoid
from oneflow._C import hardshrink
from oneflow._C import hardswish
from oneflow._C import leaky_relu
from oneflow._C import elu
from oneflow._C import celu
from oneflow._C import selu
from oneflow._C import sigmoid
from oneflow._C import softshrink
from oneflow._C import prelu
from oneflow._C import gelu_with_approximate as gelu
from oneflow._C import glu
from oneflow._C import logsigmoid
from oneflow._C import log_softmax
from oneflow._C import softsign
from oneflow._C import softmax
from oneflow._C import softplus
from oneflow._C import tanh
from oneflow._C import threshold
from oneflow._C import silu
from oneflow._C import mish
from oneflow.nn.modules.normalization import layer_norm
from oneflow._C import dropout
from oneflow._C import smooth_l1_loss
from .functional_pad import pad
from oneflow._C import triplet_margin_loss
from oneflow._C import ctc_greedy_decoder
from oneflow._C import one_hot
from oneflow._C import normalize
from oneflow._C import cross_entropy
from oneflow._C import binary_cross_entropy_loss as binary_cross_entropy
from oneflow._C import (
    binary_cross_entropy_with_logits_loss as binary_cross_entropy_with_logits,
)
from oneflow.nn.modules.sparse import embedding
from oneflow.nn.modules.linear import linear
from oneflow.nn.modules.activation import relu6
from oneflow.nn.modules.upsampling import Upsample as upsample
from oneflow._C import unfold
from oneflow._C import fold
