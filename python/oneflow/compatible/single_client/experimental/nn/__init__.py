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
from oneflow.compatible.single_client.nn.modules.activation import (
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
from oneflow.compatible.single_client.nn.modules.adaptive_pool import AdaptiveAvgPool2d
from oneflow.compatible.single_client.nn.modules.batchnorm import (
    BatchNorm1d,
    BatchNorm2d,
)
from oneflow.compatible.single_client.nn.modules.constantpad2d import ConstantPad2d
from oneflow.compatible.single_client.nn.modules.container import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from oneflow.compatible.single_client.nn.modules.conv import Conv1d, Conv2d
from oneflow.compatible.single_client.nn.modules.dataset import (
    COCOReader,
    CoinFlip,
    CropMirrorNormalize,
    OFRecordImageDecoder,
    OFRecordImageDecoderRandomCrop,
    OfrecordRawDecoder,
    OfrecordReader,
)
from oneflow.compatible.single_client.nn.modules.deconv import ConvTranspose2d
from oneflow.compatible.single_client.nn.modules.dropout import Dropout
from oneflow.compatible.single_client.nn.modules.flatten import Flatten
from oneflow.compatible.single_client.nn.modules.instancenorm import (
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
)
from oneflow.compatible.single_client.nn.modules.linear import Identity, Linear
from oneflow.compatible.single_client.nn.modules.loss import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    CTCLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    NLLLoss,
)
from oneflow.compatible.single_client.nn.modules.normalization import (
    GroupNorm,
    LayerNorm,
)
from oneflow.compatible.single_client.nn.modules.padding import (
    ReflectionPad2d,
    ReplicationPad2d,
)
from oneflow.compatible.single_client.nn.modules.pixelshuffle import PixelShuffle
from oneflow.compatible.single_client.nn.modules.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from oneflow.compatible.single_client.nn.modules.upsampling import (
    Upsample,
    UpsamplingBilinear2d,
    UpsamplingNearest2d,
)
from oneflow.compatible.single_client.nn.modules.zeropad2d import ZeroPad2d
