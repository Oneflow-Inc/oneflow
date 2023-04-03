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
from numbers import Number
import oneflow as flow
import oneflow.framework.tensor_str as tensor_str
import oneflow._oneflow_internal.lazy_mode as lazy_mode

import numpy as np
from typing import Union

Tensor = flow._oneflow_internal.Tensor
TensorTuple = flow._oneflow_internal.TensorTuple


def _ndim(self):
    return len(self.shape)


def _backward(self, gradient=None, retain_graph=False, create_graph=False):
    if lazy_mode.is_enabled():
        assert (
            self.is_lazy
        ), "nn.Graph only accept lazy tensor to call backward() in lazy mode."
        assert (
            not retain_graph
        ), "nn.Graph donot accept 'retain_graph' argument in backward() at the moment."
        assert (
            not create_graph
        ), "nn.Graph donot accept 'create_graph' argument in backward() at the moment."
        flow._oneflow_internal.nn.graph.AddTensorAsGraphLoss(self)
    flow.autograd.backward(self, gradient, retain_graph, create_graph)


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
        return flow._C.broadcast_equal(self, other)


def _cuda(self, device: Union[int, str, flow.device] = None):
    if device is None:
        device = "cuda"
    elif isinstance(device, int):
        device = "cuda:" + str(device)
    return self.to(device=device)


def _norm(self, p=None, dim=None, keepdim=False, dtype=None):
    if type(p) == str or dim != None:
        return flow._C.norm(self, p, dim, keepdim, dtype=dtype)
    return flow._C.norm(self, p, dim, keepdim, dtype=dtype, for_norm=True)


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


def _add(self, other, *, alpha=1):
    return flow._C.add(self, other, alpha=alpha)


def _addmm(self, mat1, mat2, alpha=1, beta=1):
    return flow.addmm(self, mat1, mat2, alpha, beta)


def _add_inplace(self, other, *, alpha=1):
    return flow._C.add(self, other, alpha=alpha, inplace=True)


def _iadd(self, other):
    return self.add_(other)


def _sub_inplace(self, other):
    return flow._C.sub(self, other, inplace=True)


def _expand(self, *size):
    return flow.expand(self, *size)


def _expand_as(input, other):
    return flow.expand(input, *other.size())


def _argwhere(self):
    return flow.argwhere(self)


def _index(self):
    assert self.numel() == 1 and self.dtype in (
        flow.uint8,
        flow.int8,
        flow.int32,
        flow.int64,
        flow.bool,
    ), "Only integer tensors of a single element can be converted to an index"
    return self.numpy().item()


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


def _new_empty(
    self, *size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,
):
    return flow.new_empty(self, size, dtype, device, placement, sbp, requires_grad)


def _new_ones(
    self, *size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,
):
    return flow.new_ones(self, size, dtype, device, placement, sbp, requires_grad)


def _new_zeros(
    self, *size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,
):
    return flow.new_zeros(self, size, dtype, device, placement, sbp, requires_grad)


def _squeeze_inplace(self, dim=None):
    return flow._C.squeeze_(self, dim=dim)


def _unsqueeze_inplace(self, dim=None):
    return flow._C.unsqueeze_(self, dim=dim)


def _new_full(
    self,
    size,
    fill_value,
    dtype=None,
    device=None,
    placement=None,
    sbp=None,
    requires_grad=False,
):
    return flow.new_full(
        self, size, fill_value, dtype, device, placement, sbp, requires_grad
    )


def _argsort(self, dim=-1, descending=None):
    return flow.argsort(self, dim=dim, descending=descending)


def _uniform(self, a=0, b=1):
    return flow.nn.init.uniform_(self, a, b)


def _exponential(self, lambd=1.0, generator=None):
    return flow._C.exponential_(self, lambd, generator)


def _trunc_normal_(
    self, mean=0.0, std=1.0, a=-2.0, b=2.0,
):
    return flow.nn.init.trunc_normal_(self, mean=mean, std=std, a=a, b=b)


def _kaiming_uniform(
    self, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    return flow.nn.init.kaiming_uniform_(
        self, a=a, mode=mode, nonlinearity=nonlinearity, data_format=data_format
    )


def _kaiming_normal(
    self, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    return flow.nn.init.kaiming_normal_(
        self, a=a, mode=mode, nonlinearity=nonlinearity, data_format=data_format
    )


def _xavier_normal(self, gain=1.0):
    return flow.nn.init.xavier_normal_(self, gain=gain, data_format=data_format)


def _xavier_uniform(self, gain=1.0):
    return flow.nn.init.xavier_uniform_(self, gain=gain, data_format=data_format)


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
    return flow.nn.init.normal_(self, mean=mean, std=std)


def _copy_from_numpy_to_eager_local_tensor(eager_local_tensor, np_arr):
    assert np_arr.dtype == flow.convert_oneflow_dtype_to_numpy_dtype(
        eager_local_tensor.dtype
    )
    assert np_arr.shape == tuple(eager_local_tensor.shape)
    eager_local_tensor._copy_from_numpy(np_arr)


def _copy(self, other: Union[Tensor, np.ndarray]):
    if isinstance(other, np.ndarray):
        other = flow.from_numpy(other)
    elif not isinstance(other, Tensor):
        other = flow.tensor(other)
    other = other.to(self.dtype)
    if self.is_global:
        assert other.is_global, "Only global tensor can be assigned to global tensor."
        if not (self.sbp == other.sbp and self.placement == other.placement):
            other_cpu_placement = flow.placement("cpu", other.placement.ranks)
            other = other.to_global(placement=other_cpu_placement)
            self_cpu_placement = flow.placement("cpu", self.placement.ranks)
            other = other.to_global(placement=self_cpu_placement, sbp=self.sbp)
        flow._C.assign_local_tensor(self.to_local(), other.to_local())
    else:
        assert other.is_local, "Only local tensor can be assigned to local tensor."
        other = flow._C.broadcast_like(other, self)
        if not self.is_contiguous():
            # NOTE: slice_update support non-contiguous input tensor
            with flow.no_grad():
                self[...] = other
        else:
            flow._C.assign_local_tensor(self, other)


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


def _nms(boxes, scores, iou_threshold: float):
    return flow.nms(boxes, scores, iou_threshold)


def _nonzero(self, as_tuple=False):
    return flow.nonzero(self, as_tuple)


def _prod(self, dim=[], keepdim=False):
    return flow.prod(self, dim, keepdim)


def _masked_select(self, mask):
    return flow.masked_select(self, mask)


def _sort(self, dim: int = -1, descending: bool = False):
    return flow.sort(self, dim, descending)


def _where(self, x=None, y=None):
    return flow.where(self, x, y)


def _numpy(self):
    assert (
        not self.is_lazy
    ), "tensor.numpy() is not allowed to be called in nn.Graph.build(*args) or be called by lazy tensor."
    if self.dtype == flow.tensor_buffer:
        shapes, dtypes = self._tensor_buffer_shapes_and_dtypes
        tensors = flow.tensor_buffer_to_list_of_tensors(self, shapes, dtypes)
        return [t.numpy() for t in tensors]
    # TODO: support bfloat16 to numpy in C++
    if self.dtype == flow.bfloat16:
        self = self.to(flow.float32)
    if self.is_global:
        self_cpu_placement = flow.placement("cpu", self.placement.ranks)
        self = (
            self.to_global(placement=self_cpu_placement)
            .to_global(placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast)
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


def _cumsum(self, dim, dtype=None):
    return flow._C.cumsum(self, dim, dtype=dtype)


def _cumprod(self, dim, dtype=None):
    return flow._C.cumprod(self, dim, dtype=dtype)


def _cross(self, other, dim=None):
    return flow._C.cross(self, other, dim)


def _scatter(self, dim, index, src, *, reduce=None):
    return flow._C.scatter(self, dim, index, src, reduce=reduce, inplace=False)


def _scatter_inplace(self, dim, index, src, *, reduce=None):
    return flow._C.scatter(self, dim, index, src, reduce=reduce, inplace=True)


def _scatter_add_inplace(self, dim, index, src):
    return flow._C.scatter_add(self, dim, index, src, inplace=True)


def _contains(self, element):
    r"""Check if `element` is present in tensor

    Args:
        element (Tensor or scalar): element to be checked
            for presence in current tensor"
    """
    if isinstance(element, (flow.Tensor, Number)):
        # type hint doesn't understand the __contains__ result array
        return (element == self).any().item()  # type: ignore[union-attr]

    raise RuntimeError(
        "Tensor.__contains__ only supports Tensor or scalar, but you passed in a %s."
        % type(element)
    )


def _allclose(self, other, atol=1e-08, rtol=1e-05, equal_nan=False):
    return flow._C.allclose(self, other, atol, rtol, equal_nan)


def _index_add(self, dim, index, source, alpha=1):
    return flow._C.index_add(self, dim, index, source, alpha)


def _index_add_inplace(self, dim, index, source, alpha=1):
    return flow._C.index_add_(self, dim, index, source, alpha)


def _as_strided(self, size, stride, storage_offset=0):
    return flow._C.as_strided(self, size, stride, storage_offset)


def _as_strided_inplace(self, size, stride, storage_offset=0):
    return flow._C.as_strided_(self, size, stride, storage_offset)


def RegisterMethods():
    Tensor.ndim = property(_ndim)
    Tensor.numpy = _numpy
    Tensor.add = _add
    Tensor.add_ = _add_inplace
    Tensor.sub_ = _sub_inplace
    Tensor.backward = _backward
    Tensor.__str__ = _str
    Tensor.__repr__ = _repr
    Tensor.__contains__ = _contains
    Tensor.__bool__ = is_nonzero
    Tensor.__iadd__ = _iadd
    Tensor.addmm = _addmm
    Tensor.__format__ = _format
    Tensor.__index__ = _index
    Tensor.__float__ = _scalar_float
    Tensor.__int__ = _scalar_int
    Tensor.__array__ = _numpy
    Tensor.uniform_ = _uniform
    Tensor.exponential_ = _exponential
    Tensor.trunc_normal_ = _trunc_normal_
    Tensor.kaiming_uniform_ = _kaiming_uniform
    Tensor.kaiming_normal_ = _kaiming_normal
    Tensor.xavier_normal_ = _xavier_normal
    Tensor.xavier_uniform_ = _xavier_uniform
    Tensor.orthogonal_ = _orthogonal
    Tensor.normal_ = _normal
    Tensor.copy_ = _copy
    Tensor._meta_repr = _meta_repr
    Tensor.argsort = _argsort
    Tensor.argwhere = _argwhere
    Tensor.expand = _expand
    Tensor.expand_as = _expand_as
    Tensor.new_empty = _new_empty
    Tensor.new_ones = _new_ones
    Tensor.new_zeros = _new_zeros
    Tensor.new_full = _new_full
    Tensor.squeeze_ = _squeeze_inplace
    Tensor.unsqueeze_ = _unsqueeze_inplace
    Tensor.where = _where
    Tensor.norm = _norm
    Tensor.repeat = _repeat
    Tensor.tile = _tile
    Tensor.to = _to
    Tensor.T = property(_T)
    Tensor.masked_select = _masked_select
    Tensor.eq = _eq
    Tensor.sort = _sort
    Tensor.tolist = _tolist
    Tensor.nms = _nms
    Tensor.nonzero = _nonzero
    Tensor.prod = _prod
    Tensor.is_consistent = _is_consistent
    Tensor.to_consistent = _to_consistent
    Tensor.new_tensor = _new_tensor
    Tensor.cumsum = _cumsum
    Tensor.cumprod = _cumprod
    Tensor.cross = _cross
    Tensor.scatter = _scatter
    Tensor.scatter_ = _scatter_inplace
    Tensor.scatter_add_ = _scatter_add_inplace
    Tensor.allclose = _allclose
    Tensor.index_add = _index_add
    Tensor.index_add_ = _index_add_inplace
    Tensor.as_strided = _as_strided
    Tensor.as_strided_ = _as_strided_inplace


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        return method

    return set_tensor_op
