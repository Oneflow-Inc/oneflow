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
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.device as oneflow_device
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow.python.framework.id_util as id_util
import oneflow as flow


@oneflow_export("tensor")
class Tensor:
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
        determining_initializer=None,
        data_initializer=None,
    ):
        device = device if device is not None else oneflow_api.device("cpu", 0)
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
            data_initializer=data_initializer,
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

    def data_ptr(self):
        TODO()

    def element_size(self):
        TODO()

    def _auto_determine(func):
        def wrapped_func(*args, **kwargs):
            tensor = args[0]
            if not tensor.is_determined:
                tensor.determine()
            return func(*args, **kwargs)

        return wrapped_func

    @_auto_determine
    def numpy(self):
        parallel_conf = placement_cfg.ParallelConf()
        parallel_conf.set_device_tag(self.device.type)
        machine_id = 0
        parallel_conf.add_device_name("{}:{}".format(machine_id, self.device.index))
        return remote_blob_util.BlobObjectNumpy(
            self._local_or_consistent_tensor._blob_object, parallel_conf
        )

    def tolist(self):
        TODO()

    def backward(self):
        TODO()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "[Tensor shape={} dtype={}]".format(self.shape, self.dtype)

    def __array__(self):
        TODO()

    def __sizeof__(self):
        TODO()

    def __deepcopy__(self):
        TODO()

    def determine(self, determining_initializer=None):
        assert not self.is_determined
        if determining_initializer is None:
            determining_initializer = self._determining_initializer
        self._local_or_consistent_tensor = determining_initializer(
            self._undetermined_tensor, self
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

    def set_data_initializer(self, data_initializer):
        assert isinstance(data_initializer, initializer_conf_util.InitializerConf)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        self._undetermined_tensor.data_initializer = data_initializer

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
        data_initializer=None,
    ):
        if not isinstance(shape, oneflow_api.Size):
            if not isinstance(shape, tuple):
                shape = tuple(shape)
            shape = oneflow_api.Size(shape)
        data_initializer = (
            data_initializer
            if data_initializer is not None
            # TODO: default initializer should be an "empty" initializer like pytorch
            else flow.zeros_initializer(dtype=dtype)
        )
        device = device if device is not None else oneflow_api.device("cpu", 0)
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.retain_grad = retain_grad
        self.placement = placement
        self.sbp = sbp
        self.is_consistent = is_consistent
        self.is_lazy = is_lazy
        self.data_initializer = data_initializer

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


def _default_initializer_for_determining(undetermined_tensor, tensor):
    variable_name = id_util.UniqueStr("tensor_")
    blob = flow.get_variable(
        name=variable_name,
        shape=tuple(undetermined_tensor.shape),
        dtype=undetermined_tensor.dtype,
        initializer=undetermined_tensor.data_initializer,
    )
    determined_tensor = oneflow_api.LocalTensor(
        undetermined_tensor.shape,
        undetermined_tensor.dtype,
        undetermined_tensor.device,
        undetermined_tensor.is_lazy,
        undetermined_tensor.requires_grad,
        True,
        undetermined_tensor.retain_grad,
    )
    determined_tensor._set_blob_object(blob.blob_object)
    tensor._variable_name = variable_name
    return determined_tensor
