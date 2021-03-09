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
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api


@oneflow_export("Tensor")
class Tensor:
    def __init__(
        self,
        *shape,
        dtype=None,
        device=None,
        requires_grad=False,
        retain_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
        determining_initializer=None
    ):
        dtype = dtype if dtype is not None else oneflow_api.float32
        device = device if device is not None else oneflow_api.device("cpu")
        self._local_or_consistent_tensor = None
        self._undetermined_tensor = UndeterminedTensor(
            shape,
            dtype,
            device=device,
            requires_grad=requires_grad,
            retain_grad=retain_grad,
            placement=placement,
            sbp=sbp,
            is_consistent=is_consistent,
            is_lazy=is_lazy,
        )
        if determining_initializer is None:
            determining_initializer = _default_initializer_for_determining
        self._determining_initializer = determining_initializer

    @property
    def shape(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.shape
        else:
            return self._undetermined_tensor.shape

    @property
    def device(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.device
        else:
            return self._undetermined_tensor.device

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.is_cuda
        else:
            return self._undetermined_tensor.is_cuda

    @property
    def dtype(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.dtype
        else:
            return self._undetermined_tensor.dtype

    @property
    def data(self):
        TODO()

    @property
    def grad(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.grad
        else:
            return None

    @property
    def grad_fn(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.grad_fn
        else:
            return None

    @property
    def requires_grad(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.requires_grad
        else:
            return self._undetermined_tensor.requires_grad

    @property
    def is_leaf(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.is_leaf
        else:
            return True

    def size(self):
        return self.shape

    def dim(self, idx):
        return self.shape[idx]

    def ndimension(self):
        return self.ndim

    def get_device(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.device
        else:
            return self._undetermined_tensor.device

    def nelemenet(self):
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    def retain_grad(self):
        assert self.is_determined
        self._local_or_consistent_tensor.retain_grad()

    def data_ptr(self):
        TODO()

    def element_size(self):
        return self.dtype.bytes

    def numpy(self):
        TODO()

    def tolist(self):
        TODO()

    def backward(
        self, gradient=None, retain_graph=False, create_graph=False, inputs=None
    ):
        assert self.is_determined
        TODO()  # liyurui

    def __str__(self):
        TODO()

    def __repr__(self):
        TODO()

    def __array__(self):
        TODO()

    def __sizeof__(self):
        TODO()

    def __deepcopy__(self, memo):
        TODO()

    def determine(self, determining_initializer=None):
        assert not self.is_determined
        if determining_initializer is None:
            determining_initializer = self._determining_initializer
        self._local_or_consistent_tensor = determining_initializer(
            self._undetermined_tensor
        )
        self._undetermined_tensor = None

    @property
    def is_determined(self):
        if self._local_or_consistent_tensor is not None:
            assert self._undetermined_tensor is None
            return True
        else:
            assert self._undetermined_tensor is not None
            return False

    def set_placement(self, placement):
        assert isinstance(placement, oneflow_api.Placement)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        assert self._undetermined_tensor.device is None
        self._undetermined_tensor.placement = placement

    def set_sbp(self, sbp):
        assert isinstance(sbp, oneflow_api.Distribute)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        self._undetermined_tensor.sbp = sbp

    def set_is_consistent(self, is_consistent):
        assert isinstance(is_consistent, bool)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        self._undetermined_tensor.is_consistent = is_consistent

    def set_is_lazy(self, is_lazy):
        assert isinstance(is_lazy, bool)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        self._undetermined_tensor.is_lazy = is_lazy

    @property
    def placement(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.placement
        else:
            return self._undetermined_tensor.placement

    @property
    def is_lazy(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.is_lazy
        else:
            return self._undetermined_tensor.is_lazy

    @property
    def is_consistent(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.is_consistent
        else:
            return self._undetermined_tensor.is_consistent

    @property
    def sbp(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.sbp
        else:
            return self._undetermined_tensor.sbp


class UndeterminedTensor:
    def __init__(
        self,
        shape,
        dtype,
        device=None,
        requires_grad=False,
        retain_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
    ):
        if not isinstance(shape, oneflow_api.Size):
            if not isinstance(shape, tuple):
                shape = tuple(shape)
            shape = oneflow_api.Size(shape)
        device = device if device is not None else oneflow_api.device("cpu")
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.retain_grad = retain_grad
        self.placement = placement
        self.sbp = sbp
        self.is_consistent = is_consistent
        self.is_lazy = is_lazy

    @property
    def is_cuda(self):
        device_type = None
        if self.placement is not None:
            device_type = self.placement.device_tag
        elif self.device is not None:
            device_type = self.device.type
        else:
            raise ValueError("Neither placement nor device found.")
        return device_type == "gpu" or device_type == "cuda"


def _default_initializer_for_determining(undetermined_tensor):
    TODO()
