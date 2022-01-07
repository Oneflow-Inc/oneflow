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
import oneflow.framework.tensor_str as tensor_str
import oneflow.ops.initializer_util as initializer_util
import oneflow._oneflow_internal.lazy_mode as lazy_mode
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb

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
    return self.shape.numel()


def _numel(self):
    return self.shape.numel()


def _element_size(self):
    return self.dtype.bytes


def _backward(self, gradient=None, retain_graph=False, create_graph=False):
    if not lazy_mode.is_enabled():
        flow.autograd.backward(self, gradient, retain_graph, create_graph)
    else:
        assert (
            self.is_lazy
        ), "nn.Graph only accept lazy tensor to call backward() in lazy mode."
        assert (
            self.shape.numel() == 1
        ), " loss_tensor.backward(), loss_tensor must be a scalar in nn.Graph, please use loss_tesnor.sum() or loss_tensor.mean() to make it a scalar tensor."
        assert (
            gradient is None
        ), "nn.Graph donot accept 'gradient' argument in backward() at the moment."
        assert (
            not retain_graph
        ), "nn.Graph donot accept 'retain_graph' argument in backward() at the moment."
        assert (
            not create_graph
        ), "nn.Graph donot accept 'create_graph' argument in backward() at the moment."
        flow._oneflow_internal.nn.graph.AddTensorAsGraphLoss(self)


def _getitem(self, key):
    try:
        return flow._C.tensor_getitem(self, key)
    except IndexException as e:
        # The stop condition of for in python is IndexError,
        # so we have to catch IndexException from C++ and throw IndexError
        raise IndexError(e)


def _setitem(self, key, value):
    if self.is_consistent:
        if isinstance(value, (int, float)):
            value = flow._C.consistent_constant(
                [1],
                value,
                dtype=self.dtype,
                placement=self.placement,
                sbp=flow.sbp.broadcast,
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
            value = flow._C.constant([1], value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(device=self.device)

    flow._C.tensor_setitem(self, key, value)
    return self


def _str(self):
    return self.__repr__()


def _repr(self):
    return tensor_str._gen_tensor_str(self)


def _meta_repr(self):
    return tensor_str._gen_tensor_meta_str(self)


def _eq(self, other):
    return self.eq(other)


def _ne(self, other):
    return self.ne(other)


def _and(self, other):
    return self.logical_and(other)


def _or(self, other):
    return self.logical_or(other)


def _not(self):
    return flow._C.logical_not(self)


def _xor(self, other):
    return self.logical_xor(other)


def _contiguous(self):
    # TODO: support stride mechanism
    return self


def _norm(self, ord=None, dim=None, keepdim=False, dtype=None):
    return flow._C.norm(self, ord, dim, keepdim, dtype=dtype)


def _vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
    return flow._C.vector_norm(self, ord, dim, keepdim, dtype=dtype)


def _matrix_norm(self, ord="fro", dim=(-2, -1), keepdim=False, dtype=None):
    return flow._C.matrix_norm(self, ord, dim, keepdim, dtype=dtype)


def _transpose(self, dim0, dim1):
    return flow._C.transpose(self, dim0, dim1)


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

    """
    shape = input.shape
    if shape.numel() == 0:
        raise RuntimeError("bool value of Tensor with no values is ambiguous")
    if shape.numel() > 1:
        raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
    value = input.numpy().item()
    return bool(value)


def _gt(self, other):
    return flow.gt(self, other)


def _lt(self, other):
    return flow.lt(self, other)


def _ge(self, other):
    return flow.ge(self, other)


def _le(self, other):
    return flow.le(self, other)


def _mul(self, other):
    return flow.mul(self, other)


def _mul_(self, other):
    return flow._C.mul_(self, other)


def _rmul(self, other):
    return self.mul(other)


def _add(self, other):
    return flow.add(self, other)


def _add_inplace(self, other):
    return flow.add(self, other, inplace=True)


def _iadd(self, other):
    return self.add_(other)


def _radd(self, other):
    return flow.add(self, other)


def _sub(self, other):
    return flow.sub(self, other)


def _rsub(self, other):
    return flow.sub(other, self)


def _truediv(self, other):
    return flow.div(self, other)


def _rtruediv(self, other):
    return flow.div(other, self)


def _floor_divide(self, other):
    return flow.floor_divide(self, other)


def _floor(self):
    return flow._C.floor(self)


def _floor_inplace_(self):
    return flow._C.floor_(self)


def _neg(self):
    return flow.neg(self)


def _pow(self, b):
    return flow.pow(self, b)


def _abs(self):
    return flow.abs(self)


def _exp(self):
    return flow.exp(self)


def _expand_as(input, other):
    return flow.expand(input, *other.size())


def _acos(self):
    return flow.acos(self)


def _arccos(self):
    return flow.arccos(self)


def _acosh(self):
    return flow.acosh(self)


def _arccosh(self):
    return flow.arccosh(self)


def _atanh(self):
    return flow.atanh(self)


def _atan2(self, other):
    return flow.atan2(self, other)


def _arctanh(self):
    return flow.arctanh(self)


def _sign(self):
    return flow.sign(self)


def _sinh(self):
    return flow.sinh(self)


def _tan(self):
    return flow.tan(self)


def _gelu(self):
    return flow.gelu(self)


def _mish(self):
    return flow.mish(self)


def _sigmoid(self):
    return flow.sigmoid(self)


def _tanh(self):
    return flow.tanh(self)


def _silu(self):
    return flow.silu(self)


def _selu(self):
    return flow.selu(self)


def _softsign(self):
    return flow.softsign(self)


def _swapaxes(self, dim0, dim1):
    return flow._C.swapaxes(self, dim0, dim1)


def _cast(self, dtype):
    return flow.cast(self, dtype)


def _diag(self, diagonal=0):
    return flow.diag(self, diagonal=diagonal)


def _diagonal(self, offset=0, dim1=0, dim2=1):
    return flow._C.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)


def _log1p(self):
    return flow.log1p(self)


def _reciprocal(self):
    return flow.reciprocal(self)


def _asin(self):
    return flow.asin(self)


def _arcsin(self):
    return flow.arcsin(self)


def _asinh(self):
    return flow.asinh(self)


def _arcsinh(self):
    return flow.arcsinh(self)


def _atan(self):
    return flow.atan(self)


def _arctan(self):
    return flow.arctan(self)


def _ceil(self):
    return flow.ceil(self)


def _clamp(self, min=None, max=None):
    return flow._C.clamp(self, min=min, max=max)


def _clamp_(self, min=None, max=None):
    return flow._C.clamp_(self, min=min, max=max)


def _clip(self, min=None, max=None):
    return flow._C.clip(self, min=min, max=max)


def _clip_(self, min=None, max=None):
    return flow._C.clip_(self, min=min, max=max)


def _cos(self):
    return flow.cos(self)


def _cosh(self):
    return flow.cosh(self)


def _erf(self):
    return flow.erf(self)


def _erfc(self):
    return flow.erfc(self)


def _erfinv(self):
    return flow._C.erfinv(self)


def _erfinv_inplace(self):
    return flow._C.erfinv_(self)


def _expm1(self):
    return flow.expm1(self)


def _fmod(self, other):
    return flow.fmod(self, other)


def _flatten(self, start_dim: int = 0, end_dim: int = -1):
    return flow._C.flatten(self, start_dim=start_dim, end_dim=end_dim)


def _log(self):
    return flow.log(self)


def _minimum(self, y):
    return flow.minimum(self, y)


def _maximum(self, y):
    return flow.maximum(self, y)


def _negative(self):
    return flow._C.negative(self)


def _neg(self):
    return flow._C.negative(self)


def _rsqrt(self):
    return flow.rsqrt(self)


def _sqrt(self):
    return flow.sqrt(self)


def _square(self):
    return flow.square(self)


def _var(self, dim=None, unbiased=True, keepdim=False):
    return flow._C.var(self, dim=dim, unbiased=unbiased, keepdim=keepdim)


def _std(self, dim=None, unbiased=True, keepdim=False):
    return flow._C.std(self, dim=dim, unbiased=unbiased, keepdim=keepdim)


def _squeeze(self, dim=None):
    return flow._C.squeeze(self, dim=dim)


def _unfold(self, dimension, size, step):
    return flow._C.unfold_tensor(self, dimension=dimension, size=size, step=step)


def _narrow(self, dimension, start, length):
    return flow._C.narrow(self, dim=dimension, start=start, length=length)


def _unsqueeze(self, dim):
    return flow._C.unsqueeze(self, dim=dim)


def _permute(self, *dims):
    if len(dims) == 1:
        new_dims = dims[0]
        if isinstance(new_dims, int):
            new_dims = (new_dims,)
    else:
        new_dims = dims
    return flow._C.transpose(self, new_dims)


def _matmul(self, other):
    return flow.matmul(self, other)


def _round(self):
    return flow.round(self)


def _softplus(self):
    return flow.softplus(self)


def _tril(self, diagonal=0):
    return flow.tril(self, diagonal=diagonal)


def _triu(self, diagonal=0):
    return flow.triu(self, diagonal=diagonal)


def _relu(self, inplace=False):
    return flow.relu(self, inplace=inplace)


def _softmax(self, dim=None):
    return flow.softmax(self, dim=dim)


def _log_softmax(self, dim=None):
    return flow.log_softmax(self, dim=dim)


def _argmax(self, dim=None, keepdim=None):
    return flow.argmax(self, dim=dim, keepdim=keepdim)


def _argmin(self, dim=None, keepdim=None):
    return flow.argmin(self, dim=dim, keepdim=keepdim)


def _argsort(self, dim=None, descending=None):
    return flow.argsort(self, dim=dim, descending=descending)


def _roll(self, shifts, dims=None):
    return flow.roll(self, shifts=shifts, dims=dims)


def _bmm(self, other):
    return flow.bmm(self, other)


def _chunk(self, chunks=None, dim=None):
    return flow._C.chunk(self, chunks, dim)


def _split(self, split_size_or_sections=None, dim=None):
    return flow._C.split(self, split_size_or_sections, dim)


def _all(self, dim=None, keepdim=False):
    return flow.all(self, dim, keepdim)


def _any(self, dim=None, keepdim=False):
    return flow.any(self, dim, keepdim)


def _len(self):
    if self.dim() == 0:
        raise TypeError("len() of a 0-d tensor")
    return self.shape[0]


def _uniform(self, a=0, b=1):
    if isinstance(a, Tensor):
        assert a.ndim == 0 and a.nelement() == 1, "a must be a number or scalar tensor!"
        a = a.numpy().item()
    if isinstance(b, Tensor):
        assert b.ndim == 0 and b.nelement() == 1, "b must be a number or scalar tensor!"
        b = b.numpy().item()
    initializer_conf = flow.random_uniform_initializer(
        minval=a, maxval=b, dtype=self.dtype
    )
    return _init_by_initializer_conf(self, initializer_conf)


def _trunc_normal_(
    self, mean=0.0, std=1.0, a=-2.0, b=2.0,
):
    initializer_conf = flow.truncated_normal_initializer(mean=mean, stddev=std)
    res = _init_by_initializer_conf(self, initializer_conf)
    res = flow.clamp(res, min=a, max=b)
    return res


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
        random_seed = flow.default_generator.seed()
    shape = tuple(tensor.shape)
    initializer = initializer_util.GetInitializer(initializer_conf, random_seed, shape)

    np_arr = initializer_util.generate_values_by_initializer(
        initializer, shape, tensor.dtype
    )
    if tensor.is_consistent:
        src_tensor = flow.tensor(np_arr)
        src_tensor = src_tensor.to_consistent(
            placement=tensor.placement,
            sbp=tuple(flow.sbp.broadcast for _ in range(len(tensor.sbp))),
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
        other = other.to_consistent(placement=self.placement, sbp=self.sbp)
        flow._C.assign_local_tensor(self.to_local(), other.to_local())
    else:
        if not isinstance(other, (Tensor)):
            assert isinstance(other, np.ndarray)
            _copy_from_numpy_to_eager_local_tensor(self, other)
        else:
            flow._C.assign_local_tensor(self, other.to(device=self.device))


def _get_device(self):
    if self.device.type == "cuda":
        return self.device.index
    raise NotImplementedError("get_device is only available for GPU tensor.")


def _format(self, format_spec):
    if self.dim() == 0:
        return self.numpy().tolist().__format__(format_spec)
    return object.__format__(self, format_spec)


def _to(self, *args, **kwargs):
    return flow._C.to(self, *args, **kwargs)


def _gather(self, dim, index):
    return flow._C.dim_gather(self, dim, index, False)


def RegisterMethods():
    Tensor.__mul__ = lambda self, other: self.mul(other)
    Tensor.__rmul__ = lambda self, other: self.mul(other)
    Tensor.__add__ = lambda self, other: self.add(other)
    Tensor.__iadd__ = lambda self, other: self.add_(other)
    Tensor.__matmul__ = lambda self, other: self.matmul(other)
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
    Tensor.__and__ = _and
    Tensor.__or__ = _or
    Tensor.__xor__ = _xor
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
    Tensor.__floordiv__ = _floor_divide
    Tensor.__len__ = _len
    Tensor.__mod__ = _fmod
    Tensor.uniform_ = _uniform
    Tensor.trunc_normal_ = _trunc_normal_
    Tensor.kaiming_uniform_ = _kaiming_uniform
    Tensor.kaiming_normal_ = _kaiming_normal
    Tensor.xavier_normal_ = _xavier_normal
    Tensor.xavier_uniform_ = _xavier_uniform
    Tensor.normal_ = _normal
    Tensor.fill_ = _fill
    Tensor.copy_ = _copy
    Tensor.get_device = _get_device
    Tensor._meta_repr = _meta_repr
    Tensor.abs = _abs
    Tensor.exp = _exp
    Tensor.floor_divide = _floor_divide
    Tensor.floor = _floor
    Tensor.floor_ = _floor_inplace_
    Tensor.argmax = _argmax
    Tensor.argmin = _argmin
    Tensor.argsort = _argsort
    Tensor.acos = _acos
    Tensor.arccos = _arccos
    Tensor.acosh = _acosh
    Tensor.arccosh = _arccosh
    Tensor.atanh = _atanh
    Tensor.atan2 = _atan2
    Tensor.arctanh = _arctanh
    Tensor.sign = _sign
    Tensor.sinh = _sinh
    Tensor.tan = _tan
    Tensor.gt = _gt
    Tensor.ge = _ge
    Tensor.gelu = _gelu
    Tensor.mish = _mish
    Tensor.negative = _negative
    Tensor.neg = _neg
    Tensor.sigmoid = _sigmoid
    Tensor.tanh = _tanh
    Tensor.silu = _silu
    Tensor.selu = _selu
    Tensor.softsign = _softsign
    Tensor.cast = _cast
    Tensor.diag = _diag
    Tensor.diagonal = _diagonal
    Tensor.log1p = _log1p
    Tensor.add = _add
    Tensor.add_ = _add_inplace
    Tensor.div = _truediv
    Tensor.mul = _mul
    Tensor.mul_ = _mul_
    Tensor.reciprocal = _reciprocal
    Tensor.sub = _sub
    Tensor.asin = _asin
    Tensor.arcsin = _arcsin
    Tensor.asinh = _asinh
    Tensor.arcsinh = _arcsinh
    Tensor.atan = _atan
    Tensor.arctan = _arctan
    Tensor.ceil = _ceil
    Tensor.clamp = _clamp
    Tensor.clamp_ = _clamp_
    Tensor.clip = _clip
    Tensor.clip_ = _clip_
    Tensor.cos = _cos
    Tensor.cosh = _cosh
    Tensor.expand_as = _expand_as
    Tensor.erf = _erf
    Tensor.erfc = _erfc
    Tensor.erfinv = _erfinv
    Tensor.erfinv_ = _erfinv_inplace
    Tensor.expm1 = _expm1
    Tensor.fmod = _fmod
    Tensor.flatten = _flatten
    Tensor.log = _log
    Tensor.minimum = _minimum
    Tensor.maximum = _maximum
    Tensor.pow = _pow
    Tensor.rsqrt = _rsqrt
    Tensor.sqrt = _sqrt
    Tensor.square = _square
    Tensor.var = _var
    Tensor.std = _std
    Tensor.matmul = _matmul
    Tensor.round = _round
    Tensor.softplus = _softplus
    Tensor.tril = _tril
    Tensor.triu = _triu
    Tensor.contiguous = _contiguous
    Tensor.norm = _norm
    Tensor.vector_norm = _vector_norm
    Tensor.matrix_norm = _matrix_norm
    Tensor.transpose = _transpose
    Tensor.relu = _relu
    Tensor.softmax = _softmax
    Tensor.log_softmax = _log_softmax
    Tensor.logical_not = _not
    Tensor.roll = _roll
    Tensor.bmm = _bmm
    Tensor.chunk = _chunk
    Tensor.split = _split
    Tensor.squeeze = _squeeze
    Tensor.swapaxes = _swapaxes
    Tensor.unfold = _unfold
    Tensor.narrow = _narrow
    Tensor.unsqueeze = _unsqueeze
    Tensor.permute = _permute
    Tensor.to = _to
    Tensor.gather = _gather
    Tensor.all = _all
    Tensor.any = _any


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op
