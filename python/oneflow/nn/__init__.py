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
    CELU,
    GELU,
    GLU,
    Hardsigmoid,
    Hardshrink,
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
    Softshrink,
    Softplus,
    Tanh,
    SELU,
    SiLU,
    Softsign,
    Threshold,
)

from oneflow.nn.modules.all_reduce import AllReduce
from oneflow.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d
from oneflow.nn.modules.batchnorm_fused import (
    FusedBatchNorm1d,
    FusedBatchNorm2d,
    FusedBatchNorm3d,
)
from oneflow.nn.modules.fused_mlp import FusedMLP

from oneflow.nn.modules.container import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from oneflow.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from oneflow.nn.modules.distance import CosineSimilarity
from oneflow.nn.modules.min_max_observer import MinMaxObserver
from oneflow.nn.modules.moving_average_min_max_observer import (
    MovingAverageMinMaxObserver,
)
from oneflow.nn.modules.fake_quantization import FakeQuantization
from oneflow.nn.modules.quantization import Quantization
from oneflow.nn.modules.distributed_partial_fc_sample import (
    DistributedPariticalFCSample,
)

from oneflow.nn.modules.dataset import (
    COCOReader,
    CoinFlip,
    CropMirrorNormalize,
    OFRecordImageDecoder,
    OFRecordImageDecoderRandomCrop,
    OFRecordImageGpuDecoderRandomCropResize,
    OFRecordRawDecoder,
    OFRecordRawDecoder as OfrecordRawDecoder,
    OFRecordReader,
    OFRecordReader as OfrecordReader,
    OFRecordBytesDecoder,
    GPTIndexedBinDataReader,
    OneRecReader,
    RawReader,
)

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
    CombinedMarginLoss,
    TripletMarginLoss,
)
from oneflow.nn.modules.normalization import GroupNorm, LayerNorm
from oneflow.nn.modules.padding import (
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad2d,
    ReplicationPad2d,
    ZeroPad2d,
)
from oneflow.nn.modules.pixelshuffle import PixelShufflev2 as PixelShuffle
from oneflow.nn.modules.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
)
from oneflow.nn.modules.sparse import Embedding
from oneflow.nn.modules.upsampling import (
    Upsample,
    UpsamplingBilinear2d,
    UpsamplingNearest2d,
)
from oneflow.nn.modules.fold import Fold, Unfold

from oneflow.nn.parameter import Parameter
from oneflow.nn import utils

from . import functional

from . import parallel

from oneflow.nn.modules.rnn import (
    RNNCell,
    LSTMCell,
    GRUCell,
    RNN,
    LSTM,
    GRU,
)
