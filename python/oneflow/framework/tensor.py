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
    return flow._C.floor_divide(self, other)


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


def _mv(self, vec):
    return flow._C.mv(self, vec)


def _round(self):
    return flow.round(self)


def _softplus(self):
    return flow.softplus(self)


def _tril(self, diagonal=0):
    return flow.tril(self, diagonal=diagonal)


def _triu(self, diagonal=0):
    return flow.triu(self, diagonal=diagonal)


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
    return flow._C.fill_(self, value)


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
    # Possibility 1: self and other are tensors on the same device/placement and have the same sbp.
    if isinstance(other, Tensor):
        if self.is_global:
            assert (
                other.is_global
            ), "Only global tensor can be assigned to global tensor."
            if self.placement == other.placement and self.sbp == other.sbp:
                flow._C.assign_local_tensor(self.to_local(), other.to_local())
                return
        else:
            assert (
                not other.is_global
            ), "Only local tensor can be assigned to local tensor."
            if self.device == other.device:
                flow._C.assign_local_tensor(self, other)
                return

    # Possibility 2: `other` is a numpy array, or `self` and `other` are tensors on different devices/placements.
    # In this case, we run boxing through cpu to avoid extra gpu memory usage.
    if self.is_global:
        self_cpu_placement = flow.placement("cpu", self.placement.ranks)
        if isinstance(other, Tensor):
            other_cpu_placement = flow.placement("cpu", other.placement.ranks)
            other = other.to_global(placement=other_cpu_placement).to_global(
                placement=self_cpu_placement, sbp=self.sbp
            )
        else:
            other = flow.tensor(
                other, dtype=self.dtype, placement=self_cpu_placement, sbp=self.sbp
            )
        _copy_from_numpy_to_eager_local_tensor(
            self.to_local(), other.to_local().numpy()
        )
    else:
        if isinstance(other, Tensor):
            other = other.numpy()

        _copy_from_numpy_to_eager_local_tensor(self, other)


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


def _repeat_interleave(self, *args, **kwargs):
    return flow._C.repeat_interleave(self, *args, **kwargs)


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
        self_cpu_placement = flow.placement("cpu", self.placement.ranks)
        self = (
            self.to_global(placement=self_cpu_placement)
            .to_global(
                placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.broadcast
            )
            .to_local()
        )
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
    Tensor.ndim = property(_ndim)
    Tensor.numpy = _numpy
    Tensor.add = _add
    Tensor.add_ = _add_inplace
    Tensor.sub = _sub
    Tensor.sub_ = _sub_inplace
    Tensor.backward = _backward
    Tensor.__str__ = _str
    Tensor.__repr__ = _repr
    Tensor.__bool__ = is_nonzero
    Tensor.__iadd__ = _iadd
    Tensor.addmm = _addmm
    Tensor.__format__ = _format
    Tensor.__index__ = _index
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
    Tensor._meta_repr = _meta_repr
    Tensor.argsort = _argsort
    Tensor.argwhere = _argwhere
    Tensor.expand = _expand
    Tensor.expand_as = _expand_as
    Tensor.flip = _flip
    Tensor.new_empty = _new_empty
    Tensor.new_ones = _new_ones
    Tensor.new_zeros = _new_zeros
    Tensor.where = _where
    Tensor.norm = _norm
    Tensor.repeat = _repeat
    Tensor.repeat_interleave = _repeat_interleave
    Tensor.tile = _tile
    Tensor.split = _split
    Tensor.to = _to
    Tensor.gather = _gather
    Tensor.T = property(_T)
    Tensor.masked_select = _masked_select
    Tensor.eq = _eq
    Tensor.item = _item
    Tensor.sort = _sort
    Tensor.type_as = _type_as
    Tensor.tolist = _tolist
    Tensor.is_floating_point = _is_floating_point
    Tensor.topk = _topk
    Tensor.nms = _nms
    Tensor.nonzero = _nonzero
    Tensor.prod = _prod
    Tensor.is_consistent = _is_consistent
    Tensor.to_consistent = _to_consistent
    Tensor.new_tensor = _new_tensor
    Tensor.cumsum = _cumsum
    Tensor.cumprod = _cumprod
    Tensor.mv = _mv


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op
