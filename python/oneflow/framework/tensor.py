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
import oneflow._oneflow_internal.lazy_mode as lazy_mode
from oneflow.support.blocking import BlockingInfoContext

import numpy as np
from typing import Union


Tensor = flow._oneflow_internal.Tensor
TensorTuple = flow._oneflow_internal.TensorTuple


def _tensor_numpy(eager_local_tensor):
    assert (
        not eager_local_tensor.is_lazy
    ), "tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor."
    if eager_local_tensor.dtype == flow.tensor_buffer:
        shapes, dtypes = eager_local_tensor._tensor_buffer_shapes_and_dtypes
        tensors = flow.tensor_buffer_to_list_of_tensors(
            eager_local_tensor, shapes, dtypes
        )
        return [t.numpy() for t in tensors]
    method_name = eager_local_tensor._get_copy_mirrored_tensor_to_numpy_func_name()
    copy_to_numpy = getattr(eager_local_tensor, method_name)

    ndarray = np.empty(
        shape=tuple(eager_local_tensor.shape),
        dtype=flow.convert_oneflow_dtype_to_numpy_dtype(eager_local_tensor.dtype),
    )

    with BlockingInfoContext() as ctx:
        if ndarray.size != 0:
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
    if not lazy_mode.is_enabled():
        flow.autograd.backward(self, gradient, retain_graph, create_graph)
    else:
        assert (
            self.is_lazy
        ), "nn.Graph only accept lazy tensor to call backward() in lazy mode."
        flow._oneflow_internal.nn.graph.AddTensorAsGraphLoss(self)


def _getitem(self, key):
    try:
        return flow.F.tensor_getitem(self, key)
    except IndexException as e:
        # The stop condition of for in python is IndexError,
        # so we have to catch IndexException from C++ and throw IndexError
        raise IndexError(e)


def _setitem(self, key, value):
    if self.is_consistent:
        if isinstance(value, (int, float)):
            value = flow.F.consistent_constant(
                [1], value, self.dtype, placement=self.placement, sbp=flow.sbp.broadcast
            )
        else:
            if value.is_consistent:
                value = value.to_consistent(sbp=flow.sbp.broadcast)
                # TODO: remove these lines after asymmetric boxing is ready
                local_tensor = value.to_local()
                if local_tensor.nelement() == 0:
                    local_tensor = flow.zeros(*value.shape)
                value = local_tensor.to_consistent(
                    self.placement, sbp=flow.sbp.broadcast
                )
            else:
                value = value.to_consistent(self.placement, sbp=flow.sbp.broadcast)
    else:
        if isinstance(value, (int, float)):
            value = flow.F.constant([1], value, self.dtype, device=self.device)
        else:
            value = value.to(device=self.device)

    flow.F.tensor_setitem(self, key, value)
    return self


def _str(self):
    return self.__repr__()


def _repr(self):
    return tensor_str_util._gen_tensor_str(self)


def _meta_repr(self):
    return tensor_str_util._gen_tensor_meta_str(self)


def _eq(self, other):
    return self.eq(other)


def _ne(self, other):
    return self.ne(other)


def is_nonzero(input):
    r"""
    is_nonzero(input) -> (bool)

    Returns True if the :attr:`input` is a single element tensor which is not equal to zero
    after type conversions. i.e. not equal to ``flow.tensor([0.])`` or ``flow.tensor([0])``.

    Throws a ``RuntimeError`` if ``input.shape.numel() != 1``

    For Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.is_nonzero(flow.tensor([0.]))
        False
        >>> flow.is_nonzero(flow.tensor([1.5]))
        True
        >>> flow.is_nonzero(flow.tensor([3]))
        True
        >>> flow.is_nonzero(flow.tensor([1, 3, 5]))
        Traceback (most recent call last):
        ...
        RuntimeError: bool value of Tensor with more than one value is ambiguous
        >>> flow.is_nonzero(flow.tensor([]))
        Traceback (most recent call last):
        ...
        RuntimeError: bool value of Tensor with no values is ambiguous

    """
    shape = input.shape
    if shape.numel() == 0:
        raise RuntimeError("bool value of Tensor with no values is ambiguous")
    if shape.numel() > 1:
        raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
    value = input.numpy().item()
    return bool(value)


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
    return flow.sub(other, self)


def _truediv(self, other):
    return self.div(other)


def _rtruediv(self, other):
    return flow.div(other, self)


def _neg(self):
    return flow.neg(self)


def _pow(self, b):
    return flow.pow(self, b)


def _uniform(self, a=0, b=1):
    initializer_conf = flow.random_uniform_initializer(
        minval=a, maxval=b, dtype=self.dtype
    )
    return _init_by_initializer_conf(self, initializer_conf)


def _kaiming_uniform(
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


def _kaiming_normal(
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


def _xavier_normal(self, gain=1.0, *, data_format="NCHW"):
    assert gain == 1.0, "Only gain == 1.0 is supported now"
    initializer_conf = flow.xavier_normal_initializer(data_format=data_format)
    return _init_by_initializer_conf(self, initializer_conf)


def _xavier_uniform(self, gain=1.0, *, data_format="NCHW"):
    assert gain == 1.0, "Only gain == 1.0 is supported now"
    initializer_conf = flow.xavier_uniform_initializer(data_format=data_format)
    return _init_by_initializer_conf(self, initializer_conf)


def _normal(self, mean=0, std=1):
    initializer_conf = flow.random_normal_initializer(mean=mean, stddev=std)
    return _init_by_initializer_conf(self, initializer_conf)


def _fill(self, value):
    initializer_conf = flow.constant_initializer(value=value, dtype=self.dtype)
    return _init_by_initializer_conf(self, initializer_conf)


def _copy_from_numpy_to_eager_local_tensor(eager_local_tensor, np_arr):
    method_name = eager_local_tensor._get_copy_mirrored_tensor_from_numpy_func_name()
    copy_from_numpy = getattr(eager_local_tensor, method_name)
    assert np_arr.dtype == flow.convert_oneflow_dtype_to_numpy_dtype(
        eager_local_tensor.dtype
    )
    assert np_arr.shape == tuple(eager_local_tensor.shape)
    copy_from_numpy(np_arr)


def _init_by_initializer_conf(tensor, initializer_conf, random_seed=None):
    if random_seed is None:
        random_seed = flow.default_generator().seed()
    shape = tuple(tensor.shape)
    initializer = initializer_util.GetInitializer(initializer_conf, random_seed, shape)

    np_arr = check_point_v2.generate_values_by_initializer(
        initializer, shape, tensor.dtype
    )
    if tensor.is_consistent:
        src_tensor = flow.tensor(np_arr)
        src_tensor = src_tensor.to_consistent(
            placement=tensor.placement, sbp=flow.sbp.broadcast
        )
        tensor.copy_(src_tensor)
    else:
        _copy_from_numpy_to_eager_local_tensor(
            tensor, np_arr,
        )
    return tensor


def _copy(self, other: Union[Tensor, np.ndarray]):
    if self.is_consistent:
        assert isinstance(other, Tensor)
        assert other.is_consistent
        self[:] = other
    else:
        if isinstance(other, (Tensor)):
            src_np = other.numpy()
        else:
            assert isinstance(other, np.ndarray)
            src_np = other

        _copy_from_numpy_to_eager_local_tensor(self, src_np)


def _get_device(self):
    if self.device.type == "cuda":
        return self.device.index
    raise NotImplementedError("get_device is only available for GPU tensor.")


def _format(self, format_spec):
    if self.dim() == 0:
        return self.numpy().tolist().__format__(format_spec)
    return object.__format__(self, format_spec)


def RegisterMethods():
    Tensor.__mul__ = lambda self, other: self.mul(other)
    Tensor.__rmul__ = lambda self, other: self.mul(other)
    Tensor.__add__ = lambda self, other: self.add(other)
    Tensor.__iadd__ = lambda self, other: self.add_(other)
    Tensor.ndim = property(_ndim)
    Tensor.numpy = _tensor_numpy
    Tensor.size = _size
    Tensor.dim = _ndim
    Tensor.ndimension = _ndim
    Tensor.nelement = _nelement
    Tensor.numel = _numel
    Tensor.element_size = _element_size
    Tensor.backward = _backward
    Tensor.__getitem__ = _getitem
    Tensor.__setitem__ = _setitem
    Tensor.__str__ = _str
    Tensor.__repr__ = _repr
    Tensor.__eq__ = _eq
    Tensor.__ne__ = _ne
    Tensor.__bool__ = is_nonzero
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
    Tensor.__format__ = _format
    Tensor.uniform_ = _uniform
    Tensor.kaiming_uniform_ = _kaiming_uniform
    Tensor.kaiming_normal_ = _kaiming_normal
    Tensor.xavier_normal_ = _xavier_normal
    Tensor.xavier_uniform_ = _xavier_uniform
    Tensor.normal_ = _normal
    Tensor.fill_ = _fill
    Tensor.copy_ = _copy
    Tensor.get_device = _get_device
    Tensor._meta_repr = _meta_repr


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op


def tensor(*args, **kwargs):
    return flow._oneflow_internal.tensor(*args, **kwargs)
