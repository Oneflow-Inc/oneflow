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
from operator import mod
import os
import random
import sys
import traceback
from typing import List, Optional, Sequence, Tuple, Union
import warnings

from numpy import int64

import oneflow as flow
import oneflow._oneflow_internal._C as _C
from oneflow.framework.tensor import Tensor
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple
import oneflow.framework.id_util as id_util


class DistributedPariticalFCSample(Module):
    def __init__(self, num_sample):
        super().__init__()
        self.num_sample = num_sample
        self._op = flow.stateful_op("distributed_partial_fc_sample").Input("weight")          \
                                                                    .Input("label")           \
                                                                    .Output("mapped_label")   \
                                                                    .Output("sampled_label")  \
                                                                    .Output("sampled_weight").Build()

    def forward(self, weight, label):
        res = _C.dispatch_distributed_partial_fc_sample(
            self._op,
            weight = weight,
            label = label,
            num_sample = self.num_sample
        )
        return res


def distributed_partial_fc_sample_op(weight, label, num_sample):
    warnings.warn("the function of distributed_partial_fc_sample is deprecated,advise to use nn.DistributedPariticalFCSample", DeprecationWarning)
    return DistributedPariticalFCSample(num_sample)(weight, label)
