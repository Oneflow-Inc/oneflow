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
from oneflow.nn.modules.norm import l2_normalize
from oneflow.F import conv1d
from oneflow.F import conv2d
from oneflow.F import conv3d
from oneflow.F import avgpool_1d
from oneflow.F import avgpool_2d
from oneflow.F import avgpool_3d
from oneflow.F import maxpool_1d
from oneflow.F import maxpool_2d
from oneflow.F import maxpool_3d
from oneflow.F import adaptive_avg_pool1d
from oneflow.F import adaptive_avg_pool2d
from oneflow.F import adaptive_avg_pool3d
from oneflow.F import relu
from oneflow.F import hardtanh
from oneflow.F import hardsigmoid
from oneflow.F import hardswish
from oneflow.F import leaky_relu
from oneflow.F import elu
from oneflow.F import selu
from oneflow.F import sigmoid
from oneflow.F import prelu
from oneflow.F import gelu
from oneflow.F import log_sigmoid as logsigmoid
from oneflow.F import log_sigmoid
from oneflow.F import softsign
from oneflow.F import softmax
from oneflow.F import softplus
from oneflow.F import tanh
from oneflow.F import silu
from oneflow.F import mish
from oneflow.F import layer_norm
from oneflow.F import dropout
from oneflow.F import smooth_l1_loss
from oneflow.F import pad
from oneflow.F import upsample
from oneflow.nn.modules.one_hot import one_hot
