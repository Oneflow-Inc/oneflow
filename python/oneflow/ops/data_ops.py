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
from typing import Optional, Union, Sequence, List
import oneflow as flow
from oneflow.nn.module import Module
from oneflow.framework.tensor import _init_by_initializer_conf
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util


class DecodeRandom(Module):
    def __init__(
        self,
        batch_size: int,
        shape: Sequence[int],
        initializer: Optional[initializer_conf_util.InitializerConf],
        dtype: Optional[flow.dtype] = flow.float32,
        device: Optional[flow.device] = None,
        placement: flow.placement = None,
        sbp: Union[
            flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp]
        ] = None,
    ) -> None:
        super(DecodeRandom, self).__init__()
        assert isinstance(shape, (list, tuple))
        self.shape = (batch_size,) + tuple(shape)
        self.dtype = dtype
        self.initializer = initializer
        self.device = flow.device("cpu") if device is None else device
        self.placement = placement
        self.sbp = sbp

    def forward(self):
        data = flow.empty(
            *self.shape,
            dtype=self.dtype,
            device=self.device,
            placement=self.placement,
            sbp=self.sbp
        )
        _init_by_initializer_conf(data, self.initializer)
        return data


def decode_random(
    batch_size: int,
    shape: Sequence[int],
    initializer: Optional[initializer_conf_util.InitializerConf],
    dtype: Optional[flow.dtype] = flow.float32,
    device: Optional[flow.device] = None,
    placement: flow.placement = None,
    sbp: Union[
        flow._oneflow_internal.sbp.sbp, List[flow._oneflow_internal.sbp.sbp]
    ] = None,
):
    assert isinstance(shape, (list, tuple))
    shape = (batch_size,) + tuple(shape)
    data = flow.empty(
        *shape,
        dtype=dtype,
        device=flow.device("cpu") if device is None else device,
        placement=placement,
        sbp=sbp
    )
    _init_by_initializer_conf(data, initializer)
    return data
