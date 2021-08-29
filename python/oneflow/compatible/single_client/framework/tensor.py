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
import oneflow as flow
from oneflow._oneflow_internal.exception import IndexException
import oneflow.framework.check_point_v2 as check_point_v2
import oneflow.framework.tensor_str as tensor_str_util
import oneflow.ops.initializer_util as initializer_util

import numpy as np
from typing import Union


Tensor = flow._oneflow_internal.Tensor


def _tensor_numpy(eager_local_tensor):
    if eager_local_tensor.dtype == flow.tensor_buffer:
        shapes, dtypes = eager_local_tensor._tensor_buffer_shapes_and_dtypes
        tensors = flow.tensor_buffer_to_list_of_tensors(
            Tensor(eager_local_tensor), shapes, dtypes
        )
        return [t.numpy() for t in tensors]
    method_name = eager_local_tensor._get_copy_mirrored_tensor_to_numpy_func_name()
    copy_to_numpy = getattr(eager_local_tensor, method_name)
    ndarray = np.empty(
        tuple(eager_local_tensor.shape),
        dtype=flow.convert_oneflow_dtype_to_numpy_dtype(eager_local_tensor.dtype),
    )
    copy_to_numpy(ndarray)
    return ndarray


def _size(self, idx=None):
    if idx is None:
        return self.shape
    else:
        return self.shape[idx]


def _ndim(self):
    return len(self.shape)


def _nelement(self):
    prod = 1
    for dim in self.shape:
        prod *= dim
    return prod


def _numel(self):
    return self.nelement()


def _element_size(self):
    return self.dtype.bytes


def _backward(self, gradient=None, retain_graph=False, create_graph=False):
    flow.autograd.backward(self, gradient, retain_graph, create_graph)


def _getitem(self, key):
    try:
        return flow._C.tensor_getitem(self, key)
    except IndexException as e:
        # The stop condition of for in python is IndexError,
        # so we have to catch IndexException from C++ and throw IndexError
        raise IndexError(e)


def _setitem(self, key, value):
    if isinstance(value, (int, float)):
        value = flow._C.constant([1], value, self.dtype)
    flow._C.tensor_setitem(self, key, value)
    return self


def _str(self):
    return self.__repr__()


def _repr(self):
    return tensor_str_util._gen_tensor_str(self)


def _gt(self, other):
    return self.gt(other)


def _lt(self, other):
    return self.lt(other)


def _ge(self, other):
    return self.ge(other)


def _le(self, other):
    return self.le(other)


def _mul(self, other):
    return self.mul(other)


def _rmul(self, other):
    return self.mul(other)


def _add(self, other):
    return self.add(other)


def _iadd(self, other):
    return self.add_(other)


def _radd(self, other):
    return self.add(other)


def _sub(self, other):
    return self.sub(other)


def _rsub(self, other):
    return flow.experimental.sub(other, self)


def _truediv(self, other):
    return self.div(other)


def _rtruediv(self, other):
    return flow.experimental.div(other, self)


def _neg(self):
    return flow.experimental.neg(self)


def _pow(self, b):
    return flow.experimental.pow(self, b)


def _uniform_(self, a=0, b=1):
    initializer_conf = flow.random_uniform_initializer(
        minval=a, maxval=b, dtype=self.dtype
    )
    return _init_by_initializer_conf(self, initializer_conf)


def _kaiming_uniform_(
    self, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    initializer_conf = flow.kaiming_initializer(
        shape=self.shape,
        distribution="random_uniform",
        mode=mode,
        nonlinearity=nonlinearity,
        negative_slope=a,
        data_format=data_format,
    )
    return _init_by_initializer_conf(self, initializer_conf)


def _kaiming_normal_(
    self, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    initializer_conf = flow.kaiming_initializer(
        shape=self.shape,
        distribution="random_normal",
        mode=mode,
        nonlinearity=nonlinearity,
        negative_slope=a,
        data_format=data_format,
    )
    return _init_by_initializer_conf(self, initializer_conf)


def _xavier_normal_(self, gain=1.0, *, data_format="NCHW"):
    assert gain == 1.0, "Only gain == 1.0 is supported now"
    initializer_conf = flow.xavier_normal_initializer(data_format=data_format)
    return _init_by_initializer_conf(self, initializer_conf)


def _xavier_uniform_(self, gain=1.0, *, data_format="NCHW"):
    assert gain == 1.0, "Only gain == 1.0 is supported now"
    initializer_conf = flow.xavier_uniform_initializer(data_format=data_format)
    return _init_by_initializer_conf(self, initializer_conf)


def _normal_(self, mean=0, std=1):
    initializer_conf = flow.random_normal_initializer(mean=mean, stddev=std)
    return _init_by_initializer_conf(self, initializer_conf)


def _fill_(self, value):
    initializer_conf = flow.constant_initializer(value=value, dtype=self.dtype)
    return _init_by_initializer_conf(self, initializer_conf)


def _copy_from_numpy_to_eager_local_tensor(eager_local_tensor, np_arr):
    method_name = eager_local_tensor._get_copy_mirrored_tensor_from_numpy_func_name()
    copy_from_numpy = getattr(eager_local_tensor, method_name)
    assert np_arr.dtype == flow.convert_oneflow_dtype_to_numpy_dtype(
        eager_local_tensor.dtype
    )
    if np_arr.shape == ():
        assert tuple(eager_local_tensor.shape) == (1,)
    else:
        assert np_arr.shape == tuple(eager_local_tensor.shape)
    copy_from_numpy(np_arr)


def _init_eager_local_tensor_by_initializer_conf(
    eager_local_tensor, initializer_conf, random_seed=0
):
    shape = tuple(eager_local_tensor.shape)
    initializer = initializer_util.GetInitializer(initializer_conf, random_seed, shape)
    # initializer is None if and only if the initializer_conf is empty_initializer
    if initializer is None:
        return

    _copy_from_numpy_to_eager_local_tensor(
        eager_local_tensor,
        check_point_v2.generate_values_by_initializer(
            initializer, shape, eager_local_tensor.dtype
        ),
    )


def _init_by_initializer_conf(tensor, initializer_conf):
    if tensor.is_consistent:
        with tensor._placement_scope():
            check_point_v2.init_by_initializer_conf(
                tensor, initializer_conf, True, None
            )
    else:
        _init_eager_local_tensor_by_initializer_conf(tensor, initializer_conf)
    return tensor


def _convert_to_placement_scope(placement_or_device):
    if isinstance(placement_or_device, flow.placement):
        placement = placement_or_device
        return flow.scope.placement(
            placement.device_tag,
            list(placement.parallel_conf.device_name()),
            placement.hierarchy,
        )
    else:
        device = placement_or_device
        # TODO(jianhao): replace 0 with real machine id
        machine_id = 0
        # TODO(jianhao): support cuda in of
        if device.type == "cuda":
            device_tag = "gpu"
        else:
            device_tag = device.type
        return flow.scope.placement(
            device_tag, "{}:{}".format(machine_id, device.index), None
        )


def _placement_scope(self):
    if self.is_consistent:
        return _convert_to_placement_scope(self.placement)
    else:
        return _convert_to_placement_scope(self.device)


def _copy_(self, other: Union[Tensor, np.ndarray]):
    if isinstance(other, (Tensor, check_point_v2.FileBackendVariableBlob)):
        src_np = other.numpy()
    else:
        assert isinstance(other, np.ndarray)
        src_np = other

    _copy_from_numpy_to_eager_local_tensor(self, src_np)


def RegisterMethods():
    Tensor.__mul__ = lambda self, other: self.mul(other)
    Tensor.__rmul__ = lambda self, other: self.mul(other)
    Tensor.__add__ = lambda self, other: self.add(other)
    Tensor.__iadd__ = lambda self, other: self.add_(other)
    Tensor.ndim = _ndim
    Tensor.numpy = _tensor_numpy
    Tensor.size = _size
    Tensor.dim = _ndim
    Tensor.ndimension = _ndim
    Tensor.tolist = lambda self: self.numpy().tolist()
    Tensor.nelement = _nelement
    Tensor.numel = _numel
    Tensor.element_size = _element_size
    Tensor.backward = _backward
    Tensor.__getitem__ = _getitem
    Tensor.__setitem__ = _setitem
    Tensor.__str__ = _str
    Tensor.__repr__ = _repr
    Tensor.__gt__ = _gt
    Tensor.__lt__ = _lt
    Tensor.__ge__ = _ge
    Tensor.__le__ = _le
    Tensor.__mul__ = _mul
    Tensor.__rmul__ = _rmul
    Tensor.__add__ = _add
    Tensor.__iadd__ = _iadd
    Tensor.__radd__ = _radd
    Tensor.__sub__ = _sub
    Tensor.__rsub__ = _rsub
    Tensor.__truediv__ = _truediv
    Tensor.__rtruediv__ = _rtruediv
    Tensor.__neg__ = _neg
    Tensor.__pow__ = _pow
    Tensor.uniform_ = _uniform_
    Tensor.kaiming_uniform_ = _kaiming_uniform_
    Tensor.kaiming_normal_ = _kaiming_normal_
    Tensor.xavier_normal_ = _xavier_normal_
    Tensor.xavier_uniform_ = _xavier_uniform_
    Tensor.normal_ = _normal_
    Tensor.fill_ = _fill_
    Tensor._placement_scope = _placement_scope
    Tensor.copy_ = _copy_


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op


def tensor(*args, **kwargs):
    return flow._oneflow_internal.tensor(*args, **kwargs)
