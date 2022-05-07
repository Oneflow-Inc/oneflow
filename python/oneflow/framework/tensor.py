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
import oneflow.framework.tensor_str as tensor_str
import oneflow.ops.initializer_util as initializer_util
import oneflow._oneflow_internal.lazy_mode as lazy_mode
import oneflow.core.framework.variable_meta_info_pb2 as variable_meta_info_pb

import numpy as np
from typing import Union

Tensor = flow._oneflow_internal.Tensor
TensorTuple = flow._oneflow_internal.TensorTuple


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
        ), " loss_tensor.backward(), loss_tensor must be a scalar in nn.Graph, please use loss_tensor.sum() or loss_tensor.mean() to make it a scalar tensor."
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


def _setitem(self, key, value):
    if self.is_global:
        if isinstance(value, (int, float)):
            value = flow._C.global_constant(
                [1],
                value,
                dtype=self.dtype,
                placement=self.placement,
                sbp=flow.sbp.broadcast,
            )
        else:
            if value.is_global:
                value = value.to_global(sbp=flow.sbp.broadcast)
                # TODO: remove these lines after asymmetric boxing is ready
                local_tensor = value.to_local()
                if local_tensor.nelement() == 0:
                    local_tensor = flow.zeros(*value.shape)
                value = local_tensor.to_global(self.placement, sbp=flow.sbp.broadcast)
            else:
                value = value.to_global(self.placement, sbp=flow.sbp.broadcast)
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
    if self is None and other is None:
        return True
    elif self is None or other is None:
        return False
    else:
        return flow._C.equal(self, other)


def _ne(self, other):
    return flow._C.not_equal(self, other)


def _and(self, other):
    return flow._C.logical_and(self, other)


def _or(self, other):
    return flow._C.logical_or(self, other)


def _not(self):
    return flow._C.logical_not(self)


def _xor(self, other):
    return flow._C.logical_xor(self, other)


def _cpu(self):
    return self.to(device="cpu")


def _cuda(self, device: Union[int, str, flow.device] = None):
    if device is None:
        device = "cuda"
    elif isinstance(device, int):
        device = "cuda:" + str(device)
    return self.to(device=device)


def _norm(self, p=None, dim=None, keepdim=False, dtype=None):
    return flow._C.norm(self, p, dim, keepdim, dtype=dtype)


def _transpose(self, dim0, dim1):
    return flow._C.transpose(self, dim0, dim1)


def _permute(self, *dims):
    if len(dims) == 1:
        new_dims = dims[0]
        if isinstance(new_dims, int):
            new_dims = (new_dims,)
    else:
        new_dims = dims
    return flow._C.permute(self, new_dims)


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
    return flow._C.less(self, other)


def _ge(self, other):
    return flow.ge(self, other)


def _le(self, other):
    return flow._C.less_equal(self, other)


def _mul(self, other):
    return flow._C.mul(self, other)


def _mul_(self, other):
    return flow._C.mul_(self, other)


def _rmul(self, other):
    return self.mul(other)


def _add(self, other, *, alpha=1):
    return flow._C.add(self, other, alpha=alpha)


def _addmm(self, mat1, mat2, alpha=1, beta=1):
    return flow.addmm(self, mat1, mat2, alpha, beta)


def _add_inplace(self, other, *, alpha=1):
    return flow._C.add(self, other, alpha=alpha, inplace=True)


def _iadd(self, other):
    return self.add_(other)


def _radd(self, other):
    return flow.add(self, other)


def _sub(self, other):
    return flow._C.sub(self, other)


def _sub_inplace(self, other):
    return flow._C.sub(self, other, inplace=True)


def _rsub(self, other):
    return flow._C.sub(other, self)


def _truediv(self, other):
    return flow._C.div(self, other)


def _truediv_inplace(self, other):
    return flow._C.div_(self, other)


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
    return flow._C.pow(self, b)


def _rpow(self, b):
    return flow._C.pow(b, self)


def _abs(self):
    return flow.abs(self)


def _exp(self):
    return flow.exp(self)


def _expand(self, *size):
    return flow.expand(self, *size)


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


def _sin(self):
    return flow.sin(self)


def _sin_inplace(self):
    return flow._C.sin_(self)


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


def _amax(self, dim=None, keepdim=False):
    return flow._C.amax(self, dim=dim, keepdim=keepdim)


def _swapdims(self, dim0, dim1):
    return flow._C.swapdims(self, dim0, dim1)


def _cast(self, dtype):
    return flow.cast(self, dtype)


def _diag(self, diagonal=0):
    return flow.diag(self, diagonal=diagonal)


def _diagonal(self, offset=0, dim1=0, dim2=1):
    return flow._C.diagonal(self, offset=offset, dim1=dim1, dim2=dim2)


def _log1p(self):
    return flow.log1p(self)


def _log2(self):
    return flow._C.log2(self)


def _reciprocal(self):
    return flow.reciprocal(self)


def _asin(self):
    return flow.asin(self)


def _arcsin(self):
    return flow.arcsin(self)


def _argwhere(self):
    return flow.argwhere(self)


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


def _addcmul(self, tensor1, tensor2, *, value=1):
    return flow._C.addcmul(self, tensor1, tensor2, value=value)


def _addcmul_(self, tensor1, tensor2, *, value=1):
    return flow._C.addcmul_(self, tensor1, tensor2, value=value)


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


def _half(self):
    return flow._C.to(self, flow.float16)


def _index(self):
    assert self.numel() == 1 and self.dtype in (
        flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.bool,
    ), "Only integer tensors of a single element can be converted to an index"
    return self.numpy().item()


def _invert(self):
    if self.dtype != flow.bool:
        raise TypeError(
            "~ (operator.invert) is only implemented on integer and Boolean-type tensors"
        )
    return flow._C.logical_not(self)


def _scalar_float(self):
    assert (
        self.numel() == 1
    ), "only one element tensors can be converted to Python scalars"
    return self.numpy().astype(np.float64).item()


def _scalar_int(self):
    assert (
        self.numel() == 1
    ), "only one element tensors can be converted to Python scalars"
    return self.numpy().astype(np.int64).item()


def _flatten(self, start_dim: int = 0, end_dim: int = -1):
    return flow._C.flatten(self, start_dim=start_dim, end_dim=end_dim)


def _item(self):
    assert self.numel() == 1, "Only a Tensor with 1 element can be converted to Scalar"
    return self.numpy().item()


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


def _new_empty(
    self, *size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,
):
    return flow.new_empty(self, size, dtype, device, placement, sbp, requires_grad)


def _new_ones(
    self,
    size=None,
    dtype=None,
    device=None,
    placement=None,
    sbp=None,
    requires_grad=False,
):
    return flow.new_ones(self, size, dtype, device, placement, sbp, requires_grad)


def _new_zeros(
    self, *size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,
):
    return flow.new_zeros(self, size, dtype, device, placement, sbp, requires_grad)


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


def _to_local(self):
    return flow.to_local(self)


def _relu(self):
    return flow._C.relu(self)


def _relu_inplace(self):
    return flow.relu(self, inplace=True)


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


def _split(self, split_size_or_sections=None, dim=0):
    return flow._C.split(self, split_size_or_sections, dim)


def _unbind(self, dim=0):
    return flow._C.unbind(self, dim)


def _all(self, dim=[], keepdim=False):
    return flow.all(self, dim, keepdim)


def _any(self, dim=[], keepdim=False):
    return flow.any(self, dim, keepdim)


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


def _orthogonal(self, gain=1.0):
    if self.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = self.shape[0]
    cols = np.prod(self.shape[1:])
    flattened = np.random.normal(0.0, 1.0, size=(rows, cols))
    if rows < cols:
        flattened = flattened.T
    # TODO
    q, r = np.linalg.qr(flattened)
    d = np.diag(r, 0)
    d = np.sign(d)
    q *= d
    if rows < cols:
        q = q.T
    self = gain * flow.tensor(q.reshape(self.shape))
    return self


def _normal(self, mean=0, std=1):
    if self.is_global:
        src_tensor = flow.normal(mean, std, self.shape)
        src_tensor = src_tensor.to_global(
            placement=self.placement,
            sbp=tuple(flow.sbp.broadcast for _ in range(len(self.sbp))),
        )
        self.copy_(src_tensor)
        return self
    else:
        return flow.normal(
            mean,
            std,
            self.size(),
            out=self,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )


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
        random_seed = flow.default_generator.initial_seed()
    shape = tuple(tensor.shape)
    initializer = initializer_util.GetInitializer(initializer_conf, random_seed, shape)

    np_arr = initializer_util.generate_values_by_initializer(
        initializer, shape, tensor.dtype
    )
    if tensor.is_global:
        src_tensor = flow.tensor(np_arr)
        src_tensor = src_tensor.to_global(
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
    if self.is_global:
        if not isinstance(other, Tensor):
            assert isinstance(other, np.ndarray)
            other = flow.tensor(
                other, dtype=self.dtype, placement=self.placement, sbp=self.sbp
            )
        else:
            assert other.is_global
            other = other.to_global(placement=self.placement, sbp=self.sbp)
        flow._C.assign_local_tensor(self.to_local(), other.to_local())
    else:
        if not isinstance(other, (Tensor)):
            assert isinstance(other, np.ndarray)
            _copy_from_numpy_to_eager_local_tensor(self, other)
        else:
            flow._C.assign_local_tensor(self, other.to(device=self.device))


def _flip(self, dims):
    return flow.flip(self, dims)


def _in_top_k(self, predictions, k):
    return flow._C.in_top_k(self, predictions, k)


def _index_select(self, dim, index):
    return flow.index_select(self, dim, index)


def _get_device(self):
    if self.device.type == "cuda":
        return self.device.index
    raise NotImplementedError("get_device is only available for GPU tensor.")


def _format(self, format_spec):
    if self.dim() == 0:
        return self.numpy().tolist().__format__(format_spec)
    return object.__format__(self, format_spec)


def _to(self, *args, **kwargs):
    new_args = list()
    # If device is single int, replace it with flow.device("cuda:{device}")
    if len(args) > 0 and isinstance(args[0], int):
        new_args.append(flow.device(f"cuda:{args[0]}"))
        for i in range(1, len(args)):
            new_args.append(args[i])
    else:
        new_args = args
    if ("device" in kwargs) and isinstance(kwargs["device"], int):
        kwargs["device"] = flow.device(f"cuda:{kwargs['device']}")
    return flow._C.to(self, *new_args, **kwargs)


def _local_to_global(self, placement=None, sbp=None, *, check_meta=True):
    return flow.local_to_global(self, placement, sbp, check_meta)


def _global_to_global(
    self, placement=None, sbp=None, *, grad_sbp=None, check_meta=False
):
    return flow.global_to_global(self, placement, sbp, grad_sbp, check_meta)


def _to_global(self, placement=None, sbp=None, **kwargs):
    return flow.to_global(self, placement, sbp, **kwargs)


def _to_local(self):
    return flow.to_local(self)


def _tolist(self):
    if self.numel() == 1 and self.ndim == 0:
        return self.item()
    return self.numpy().tolist()


def _gather(self, dim, index):
    return flow._C.dim_gather(self, dim, index, False)


def _repeat(self, *sizes):
    if len(sizes) == 1:
        new_sizes = sizes[0]
        if isinstance(new_sizes, int):
            new_sizes = (new_sizes,)
    else:
        new_sizes = sizes
    return flow._C.repeat(self, new_sizes)


def _tile(self, *dims):
    if len(dims) == 1:
        new_dims = dims[0]
        if isinstance(new_dims, int):
            new_dims = (new_dims,)
    else:
        new_dims = dims
    return flow._C.tile(self, new_dims)


def _T(self):
    return flow._C.T(self)


def _t(self):
    return flow._C.t(self)


def _topk(self, k, dim: int = None, largest: bool = True, sorted: bool = True):
    return flow.topk(self, k, dim, largest, sorted)


def _nms(boxes, scores, iou_threshold: float):
    return flow.nms(boxes, scores, iou_threshold)


def _nonzero(self, as_tuple=False):
    return flow.nonzero(self, as_tuple)


def _max(self, *args, **kwargs):
    return flow.max(self, *args, **kwargs)


def _min(self, *args, **kwargs):
    return flow.min(self, *args, **kwargs)


def _median(self, *args, **kwargs):
    return flow.median(self, *args, **kwargs)


def _sum(self, dim=[], keepdim=False):
    return flow.sum(self, dim, keepdim)


def _mean(self, dim=[], keepdim=False):
    return flow.mean(self, dim, keepdim)


def _prod(self, dim=[], keepdim=False):
    return flow.prod(self, dim, keepdim)


def _masked_fill(self, mask, fill_value):
    return flow.masked_fill(self, mask, fill_value)


def _masked_select(self, mask):
    return flow.masked_select(self, mask)


def _reshape(self, *shape):
    if len(shape) == 1:
        new_shape = shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
    else:
        new_shape = shape
    return flow._C.reshape(self, new_shape)


def _reshape_as(self, other):
    return _reshape(self, other.size())


def _view(self, *shape):
    if len(shape) == 1:
        new_shape = shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
    else:
        new_shape = shape
    return flow._C.view(self, new_shape)


def _view_as(self, other):
    return _view(self, *other.size())


def _sort(self, dim: int = -1, descending: bool = False):
    return flow.sort(self, dim, descending)


def _type_as(self, target):
    return self.to(dtype=target.dtype)


def _int(self):
    return self.to(dtype=flow.int32)


def _long(self):
    return self.to(dtype=flow.int64)


def _float(self):
    return self.to(dtype=flow.float32)


def _double(self):
    return self.to(dtype=flow.float64)


def _where(self, x=None, y=None):
    return flow.where(self, x, y)


def _is_floating_point(self):
    return flow.is_floating_point(self)


def _numpy(self):
    assert (
        not self.is_lazy
    ), "tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor."
    if self.dtype == flow.tensor_buffer:
        shapes, dtypes = self._tensor_buffer_shapes_and_dtypes
        tensors = flow.tensor_buffer_to_list_of_tensors(self, shapes, dtypes)
        return [t.numpy() for t in tensors]
    if self.is_global:
        self = self.to_global(
            placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.broadcast
        ).to_local()
    assert self.is_local
    if self.device != flow.device("cpu"):
        self = self.cpu()
    return self.to_numpy()


def zero_(self):
    self.zero_()
    return self


def _is_consistent(self):
    raise RuntimeError(".is_consistent has been removed, please use .is_global instead")


def _to_consistent(self, *args, **kwargs):
    raise RuntimeError(".to_consistent has been removed, please use .to_global instead")


def _isnan(self):
    return flow.isnan(self)


def _isinf(self):
    return flow.isinf(self)


def _new_tensor(
    self, data, dtype=None, device=None, requires_grad=False, placement=None, sbp=None
):
    if dtype is None:
        dtype = self.dtype
    if self.is_local:
        assert (
            placement is None and sbp is None
        ), "self is local tensor, placement and sbp are expected to be None."
        if device is None:
            device = self.device
        return flow.tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
    else:
        assert device is None, "self is global tensor, device is expected to be None."
        if placement is None:
            placement = self.placement
        if sbp is None:
            sbp = self.sbp
        return flow.tensor(
            data, dtype=dtype, placement=placement, sbp=sbp, requires_grad=requires_grad
        )


def _amin(self, dim=None, keepdim=False):
    return flow._C.amin(self, dim=dim, keepdim=keepdim)


def _byte(self):
    return flow._C.to(self, flow.uint8)


def _cumsum(self, dim, dtype=None):
    return flow._C.cumsum(self, dim, dtype=dtype)


def _cumprod(self, dim, dtype=None):
    return flow._C.cumprod(self, dim, dtype=dtype)


def RegisterMethods():
    Tensor.__mul__ = lambda self, other: self.mul(other)
    Tensor.__rmul__ = lambda self, other: self.mul(other)
    Tensor.__add__ = lambda self, other: self.add(other)
    Tensor.__iadd__ = lambda self, other: self.add_(other)
    Tensor.__matmul__ = lambda self, other: self.matmul(other)
    Tensor.byte = _byte
    Tensor.ndim = property(_ndim)
    Tensor.numpy = _numpy
    Tensor.size = _size
    Tensor.dim = _ndim
    Tensor.ndimension = _ndim
    Tensor.nelement = _nelement
    Tensor.numel = _numel
    Tensor.element_size = _element_size
    Tensor.backward = _backward
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
    Tensor.addmm = _addmm
    Tensor.__sub__ = _sub
    Tensor.__rsub__ = _rsub
    Tensor.__truediv__ = _truediv
    Tensor.__rtruediv__ = _rtruediv
    Tensor.__neg__ = _neg
    Tensor.__pow__ = _pow
    Tensor.__rpow__ = _rpow
    Tensor.__format__ = _format
    Tensor.__floordiv__ = _floor_divide
    Tensor.__mod__ = _fmod
    Tensor.__index__ = _index
    Tensor.__invert__ = _invert
    Tensor.__float__ = _scalar_float
    Tensor.__int__ = _scalar_int
    Tensor.__array__ = _numpy
    Tensor.uniform_ = _uniform
    Tensor.trunc_normal_ = _trunc_normal_
    Tensor.kaiming_uniform_ = _kaiming_uniform
    Tensor.kaiming_normal_ = _kaiming_normal
    Tensor.xavier_normal_ = _xavier_normal
    Tensor.xavier_uniform_ = _xavier_uniform
    Tensor.orthogonal_ = _orthogonal
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
    Tensor.argwhere = _argwhere
    Tensor.acos = _acos
    Tensor.arccos = _arccos
    Tensor.acosh = _acosh
    Tensor.amin = _amin
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
    Tensor.log2 = _log2
    Tensor.add = _add
    Tensor.add_ = _add_inplace
    Tensor.addcmul = _addcmul
    Tensor.addcmul_ = _addcmul_
    Tensor.div = _truediv
    Tensor.div_ = _truediv_inplace
    Tensor.mul = _mul
    Tensor.mul_ = _mul_
    Tensor.reciprocal = _reciprocal
    Tensor.sub = _sub
    Tensor.sub_ = _sub_inplace
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
    Tensor.cpu = _cpu
    Tensor.cuda = _cuda
    Tensor.expand = _expand
    Tensor.expand_as = _expand_as
    Tensor.erf = _erf
    Tensor.erfc = _erfc
    Tensor.erfinv = _erfinv
    Tensor.erfinv_ = _erfinv_inplace
    Tensor.expm1 = _expm1
    Tensor.fmod = _fmod
    Tensor.flatten = _flatten
    Tensor.flip = _flip
    Tensor.in_top_k = _in_top_k
    Tensor.index_select = _index_select
    Tensor.log = _log
    Tensor.minimum = _minimum
    Tensor.maximum = _maximum
    Tensor.new_empty = _new_empty
    Tensor.new_ones = _new_ones
    Tensor.new_zeros = _new_zeros
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
    Tensor.where = _where
    Tensor.norm = _norm
    Tensor.transpose = _transpose
    Tensor.permute = _permute
    Tensor.local_to_global = _local_to_global
    Tensor.global_to_global = _global_to_global
    Tensor.to_global = _to_global
    Tensor.relu = _relu
    Tensor.relu_ = _relu_inplace
    Tensor.softmax = _softmax
    Tensor.log_softmax = _log_softmax
    Tensor.logical_and = _and
    Tensor.logical_or = _or
    Tensor.logical_not = _not
    Tensor.logical_xor = _xor
    Tensor.roll = _roll
    Tensor.bmm = _bmm
    Tensor.chunk = _chunk
    Tensor.repeat = _repeat
    Tensor.tile = _tile
    Tensor.split = _split
    Tensor.unbind = _unbind
    Tensor.squeeze = _squeeze
    Tensor.swapaxes = _swapaxes
    Tensor.amax = _amax
    Tensor.swapdims = _swapdims
    Tensor.unfold = _unfold
    Tensor.narrow = _narrow
    Tensor.unsqueeze = _unsqueeze
    Tensor.to = _to
    Tensor.half = _half
    Tensor.gather = _gather
    Tensor.all = _all
    Tensor.any = _any
    Tensor.T = property(_T)
    Tensor.t = _t
    Tensor.masked_fill = _masked_fill
    Tensor.masked_select = _masked_select
    Tensor.eq = _eq
    Tensor.ne = _ne
    Tensor.item = _item
    Tensor.lt = _lt
    Tensor.le = _le
    Tensor.to_local = _to_local
    Tensor.reshape = _reshape
    Tensor.reshape_as = _reshape_as
    Tensor.view = _view
    Tensor.view_as = _view_as
    Tensor.sort = _sort
    Tensor.type_as = _type_as
    Tensor.tolist = _tolist
    Tensor.int = _int
    Tensor.long = _long
    Tensor.float = _float
    Tensor.double = _double
    Tensor.is_floating_point = _is_floating_point
    Tensor.topk = _topk
    Tensor.nms = _nms
    Tensor.nonzero = _nonzero
    Tensor.max = _max
    Tensor.min = _min
    Tensor.median = _median
    Tensor.sum = _sum
    Tensor.mean = _mean
    Tensor.prod = _prod
    Tensor.sin = _sin
    Tensor.sin_ = _sin_inplace
    Tensor.is_consistent = _is_consistent
    Tensor.to_consistent = _to_consistent
    Tensor.isnan = _isnan
    Tensor.isinf = _isinf
    Tensor.new_tensor = _new_tensor
    Tensor.cumsum = _cumsum
    Tensor.cumprod = _cumprod


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op
