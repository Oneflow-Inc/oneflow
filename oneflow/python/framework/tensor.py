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
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow._oneflow_internal
import numpy as np
import inspect
from typing import Union
import oneflow._oneflow_internal.oneflow.core.job.placement as placement_cfg
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.check_point_v2 as check_point_v2
from oneflow.python.framework.function_util import global_function_or_identity
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow.python.framework.ofblob as ofblob_util
import oneflow.python.lib.core.async_util as async_util
import oneflow.python.ops.initializer_util as initializer_util
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.tensor_str as tensor_str_util
import oneflow as flow


def register_local_tensor_method(name=None):
    def decorator(method):
        if name is None:
            op_name = method.__name__
        else:
            op_name = name
        setattr(oneflow._oneflow_internal.LocalTensor, op_name, method)
        return method

    return decorator


@register_local_tensor_method("numpy")
def _local_tensor_numpy(eager_local_tensor):
    if eager_local_tensor.dtype == flow.tensor_buffer:
        shapes, dtypes = eager_local_tensor._tensor_buffer_shapes_and_dtypes
        tensors = flow.experimental.tensor_buffer_to_list_of_tensors(
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


@register_local_tensor_method("copy_")
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


@register_local_tensor_method("_init_by_initializer_conf")
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


@oneflow_export("tensor")
def construct_tensor(
    data,
    dtype=None,
    device=None,
    requires_grad=False,
    placement=None,
    sbp=None,
    is_consistent=False,
    is_lazy=False,
):
    if _is_scalar(data) or _input_args_is_data(data):
        if (
            not _input_args_is_numpy(data)
            and dtype is None
            and _input_dtype_is_float(data)
        ):
            dtype = flow.float32
        data = np.array(data)
        if dtype is None:
            dtype = dtype_util.convert_numpy_dtype_to_oneflow_dtype(data.dtype)
        return Tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            placement=placement,
            sbp=sbp,
            is_consistent=is_consistent,
            is_lazy=is_lazy,
        )
    else:
        raise TypeError("Construction error, invalid combination of arguments")


@oneflow_export("Tensor")
class Tensor:
    def __init__(
        self,
        *args,
        dtype=None,
        device=None,
        requires_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
        data_initializer=None,
        determining_initializer=None,
    ):
        assert len(args) > 0
        dtype = dtype if dtype is not None else oneflow._oneflow_internal.float32
        if isinstance(device, str):
            device = flow.device(device)
        if placement is None:
            device = (
                device
                if device is not None
                else oneflow._oneflow_internal.device("cpu")
            )
        if _input_args_is_tensor(*args):
            self._local_or_consistent_tensor = flow.to(
                *args, device=args[0].device, dtype=args[0].dtype, copy=True
            )
            self._undetermined_tensor = None
        elif _input_args_is_consistent_or_local(*args):
            self._local_or_consistent_tensor = args[0]
            self._undetermined_tensor = None
        elif _input_args_is_data(*args):
            self._local_or_consistent_tensor = None
            self._construct_with_data(
                *args,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                placement=placement,
                sbp=sbp,
                is_consistent=is_consistent,
                is_lazy=is_lazy,
            )
        elif _input_args_is_shape(*args):
            shape = args
            self._local_or_consistent_tensor = None
            self._undetermined_tensor = UndeterminedTensor(
                shape,
                dtype,
                device=device,
                requires_grad=requires_grad,
                placement=placement,
                sbp=sbp,
                is_consistent=is_consistent,
                is_lazy=is_lazy,
                data_initializer=data_initializer,
            )
            if determining_initializer is None:
                determining_initializer = _default_initializer_for_determining
            self._determining_initializer = determining_initializer
        else:
            # Maybe some other arguments to be supported, reported as error for now
            raise TypeError("new() received an invalid combination of arguments")

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

    @register_local_tensor_method("ndim")
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

    # internal decorator
    def _auto_determine(func):
        def wrapped_func(*args, **kwargs):
            tensor = args[0]
            if not tensor.is_determined:
                tensor.determine()
            return func(*args, **kwargs)

        return wrapped_func

    @property
    @_auto_determine
    def data(self):
        if self._local_or_consistent_tensor is not None:
            return flow.Tensor(self._local_or_consistent_tensor.data)
        else:
            return None

    @property
    def grad(self):
        if self._local_or_consistent_tensor is not None:
            if self._local_or_consistent_tensor.grad is not None:
                return flow.Tensor(self._local_or_consistent_tensor.grad)
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

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if self._local_or_consistent_tensor is not None:
            self._local_or_consistent_tensor.requires_grad = requires_grad
        else:
            self._undetermined_tensor.requires_grad = requires_grad

    @register_local_tensor_method()
    def size(self, idx=None):
        if idx is None:
            return self.shape
        else:
            return self.shape[idx]

    @register_local_tensor_method()
    def dim(self):
        return self.ndim

    @register_local_tensor_method()
    def ndimension(self):
        return self.ndim

    @_auto_determine
    def detach(self):
        if self._local_or_consistent_tensor is not None:
            return flow.Tensor(self._local_or_consistent_tensor.detach())
        else:
            return None

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad

    def get_device(self):
        if self._local_or_consistent_tensor is not None:
            return self._local_or_consistent_tensor.device
        else:
            return self._undetermined_tensor.device

    @register_local_tensor_method()
    def nelement(self):
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @register_local_tensor_method()
    def numel(self):
        return self.nelement()

    def retain_grad(self):
        assert self.is_determined
        self._local_or_consistent_tensor.retain_grad()

    def data_ptr(self):
        TODO()

    def element_size(self):
        return self.dtype.bytes

    @_auto_determine
    def numpy(self):
        internal_tensor = self._local_or_consistent_tensor
        if not internal_tensor.is_lazy and not internal_tensor.is_consistent:
            return _local_tensor_numpy(internal_tensor)

        raise NotImplementedError()

    @register_local_tensor_method()
    def tolist(self):
        return self.numpy().tolist()

    @_auto_determine
    @register_local_tensor_method()
    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        flow.autograd.backward(self, gradient, retain_graph, create_graph)

    @register_local_tensor_method()
    def _transform_ellipsis_type(self, key):
        d = self.ndim - len(key)  # exclude all Ellipsis type
        new_key = list()
        for k in key:
            if isinstance(k, type(Ellipsis)):
                new_key.append(slice(None, None, None))
                while d > 0:
                    new_key.append(slice(None, None, None))
                    d -= 1
            else:
                new_key.append(k)
        return tuple(new_key)

    @register_local_tensor_method()
    def _get_slice_obj(self, key):
        def get_or_default(x, default):
            return x if x is not None else default

        def get_canonical_index(index, length, *, start=0):
            if index < 0:
                index += length
            return max(min(index, length), start)

        def get_slice_if_int(x):
            if isinstance(x, slice):
                return x
            return slice(x, x + 1)

        if isinstance(key, tuple):
            assert all(isinstance(x, (slice, int)) for x in key)
        else:
            assert isinstance(key, (slice, int))
            key = (key,)

        key = list(map(get_slice_if_int, key))

        assert len(key) <= len(self.shape)
        for i in range(len(key), len(self.shape)):
            key += (slice(None, None, None),)

        starts = [
            get_canonical_index(get_or_default(x.start, 0), self.shape[i])
            for i, x in enumerate(key)
        ]
        stops = [
            get_canonical_index(
                get_or_default(x.stop, self.shape[i]), self.shape[i], start=starts[i]
            )
            for i, x in enumerate(key)
        ]
        steps = [get_or_default(x.step, 1) for x in key]
        assert all(x > 0 for x in steps)
        # np.abs is for compatibility of negative steps in the future
        shape = (np.abs(np.array(stops) - np.array(starts)) - 1) // np.abs(
            np.array(steps)
        ) + 1
        shape = shape.tolist()
        return starts, stops, steps, shape

    @_auto_determine
    @register_local_tensor_method()
    def __getitem__(self, key):
        # TODO: support inplace __getitem__
        assert (
            isinstance(key, int) or isinstance(key, tuple) or isinstance(key, slice)
        ), "Unsupported key type!"

        squeeze_dims = None
        if isinstance(key, tuple):
            key = self._transform_ellipsis_type(key)
            squeeze_dims = list(
                filter(lambda idx: isinstance(key[idx], int), range(len(key)))
            )
        elif isinstance(key, int):
            squeeze_dims = [0]

        start, stop, step, _ = self._get_slice_obj(key)
        res = flow.experimental.slice(self, list(zip(start, stop, step)))
        return res.squeeze(dim=squeeze_dims)

    @_auto_determine
    @register_local_tensor_method()
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = self._transform_ellipsis_type(key)
        start, stop, step, shape = self._get_slice_obj(key)
        if isinstance(value, (int, float)):
            scalar = value
            value = flow.Tensor(*shape)
            value.fill_(scalar)

        flow.experimental.tmp.logical_slice_assign(
            self, value, list(zip(start, stop, step))
        )
        return self

    @register_local_tensor_method()
    def __str__(self):
        return self.__repr__()

    @register_local_tensor_method()
    def __repr__(self):
        return tensor_str_util._gen_tensor_str(self)

    @register_local_tensor_method()
    def __gt__(self, other):
        return self.gt(other)

    @register_local_tensor_method()
    def __lt__(self, other):
        return self.lt(other)

    def __array__(self):
        TODO()

    def __sizeof__(self):
        TODO()

    def __deepcopy__(self, memo):
        TODO()

    @register_local_tensor_method()
    def __mul__(self, other):
        return self.mul(other)

    @register_local_tensor_method()
    def __rmul__(self, other):
        return self.mul(other)

    @register_local_tensor_method()
    def __add__(self, other):
        return self.add(other)

    @register_local_tensor_method()
    def __radd__(self, other):
        return self.add(other)

    @register_local_tensor_method()
    def __sub__(self, other):
        return self.sub(other)

    @register_local_tensor_method()
    def __rsub__(self, other):
        return flow.experimental.sub(other, self)

    @register_local_tensor_method()
    def __truediv__(self, other):
        return self.div(other)

    @register_local_tensor_method()
    def __rtruediv__(self, other):
        return flow.experimental.div(other, self)

    @register_local_tensor_method()
    def __neg__(self):
        return flow.experimental.neg(self)

    @register_local_tensor_method()
    def __pow__(self, b):
        return flow.experimental.pow(self, b)

    def _determine_if_needed(self, determining_initializer=None):
        if not self.is_determined:
            self.determine(determining_initializer)

    def determine(self, determining_initializer=None):
        assert not self.is_determined
        if determining_initializer is None:
            determining_initializer = self._determining_initializer
        self._local_or_consistent_tensor = determining_initializer(self)
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
        assert isinstance(placement, flow.placement)
        assert self._local_or_consistent_tensor is None
        assert self._undetermined_tensor is not None
        self._undetermined_tensor.placement = placement
        self._undetermined_tensor.device = None

    def set_sbp(self, sbp):
        assert isinstance(sbp, oneflow._oneflow_internal.Distribute)
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

    @register_local_tensor_method()
    def uniform_(self, a=0, b=1):
        initializer_conf = flow.random_uniform_initializer(
            minval=a, maxval=b, dtype=self.dtype
        )
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def kaiming_uniform_(
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
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def kaiming_normal_(
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
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def xavier_normal_(self, gain=1.0, *, data_format="NCHW"):
        assert gain == 1.0, "Only gain == 1.0 is supported now"
        initializer_conf = flow.xavier_normal_initializer(data_format=data_format)
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def xavier_uniform_(self, gain=1.0, *, data_format="NCHW"):
        assert gain == 1.0, "Only gain == 1.0 is supported now"
        initializer_conf = flow.xavier_uniform_initializer(data_format=data_format)
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def normal_(self, mean=0, std=1):
        initializer_conf = flow.random_normal_initializer(mean=mean, stddev=std)
        return self._init_by_initializer_conf(initializer_conf)

    @register_local_tensor_method()
    def fill_(self, value):
        initializer_conf = flow.constant_initializer(value=value, dtype=self.dtype)
        return self._init_by_initializer_conf(initializer_conf)

    @_auto_determine
    def zeros_(self):
        internal_tensor = self._local_or_consistent_tensor
        if internal_tensor.is_lazy:
            TODO()
        if internal_tensor.is_consistent:
            TODO()
        internal_tensor.zeros_()

    @_auto_determine
    @register_local_tensor_method()
    def register_hook(self, hook):
        assert self.is_leaf, "register_hook only supports leaf tensor for now"
        assert (
            self.requires_grad
        ), "register_hook only supports tensor with requires_grad=True"

        def hook_returning_determined_tensor(grad):
            new_grad = hook(grad)
            if isinstance(new_grad, Tensor) and not new_grad.is_determined:
                new_grad.determine()
                new_grad = new_grad._local_or_consistent_tensor
            return new_grad

        self._local_or_consistent_tensor._register_hook(
            hook_returning_determined_tensor
        )

    @_auto_determine
    def copy_(self, other: Union["Tensor", np.ndarray]):
        internal_tensor = self._local_or_consistent_tensor
        if internal_tensor.is_lazy:
            TODO()
        if internal_tensor.is_consistent:
            TODO()

        if isinstance(other, (Tensor, check_point_v2.FileBackendVariableBlob)):
            src_np = other.numpy()
        else:
            assert isinstance(other, np.ndarray)
            src_np = other

        _copy_from_numpy_to_eager_local_tensor(internal_tensor, src_np)

    def _init_by_initializer_conf(self, initializer_conf):
        if self.is_determined:
            if self.is_consistent:
                with self._placement_scope():
                    check_point_v2.init_by_initializer_conf(
                        self, initializer_conf, True, None
                    )
            else:
                _init_eager_local_tensor_by_initializer_conf(
                    self._local_or_consistent_tensor, initializer_conf
                )
        else:
            self.set_data_initializer(initializer_conf)
        return self

    def _placement_scope(self):
        if self.is_consistent:
            return _convert_to_placement_scope(self.placement)
        else:
            return _convert_to_placement_scope(self.device)

    def _construct_with_data(
        self,
        *args,
        dtype=None,
        device=None,
        requires_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
    ):
        numpy_data = None
        if _input_args_is_tuple_or_list(*args):
            numpy_data = np.array(args[0])
        elif _input_args_is_numpy(*args):
            numpy_data = args[0]
        numpy_data = numpy_data.astype(flow.convert_oneflow_dtype_to_numpy_dtype(dtype))
        shape = oneflow._oneflow_internal.Size(tuple(numpy_data.shape))
        self._determining_initializer = _numpy_initializer_for_determining
        self._undetermined_tensor = UndeterminedTensor(
            shape,
            dtype,
            device=device,
            requires_grad=requires_grad,
            placement=placement,
            sbp=sbp,
            is_consistent=is_consistent,
            is_lazy=is_lazy,
            numpy_data=numpy_data,
        )


class UndeterminedTensor:
    def __init__(
        self,
        shape,
        dtype,
        device=None,
        requires_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
        data_initializer=None,
        numpy_data=None,
    ):
        if not isinstance(shape, oneflow._oneflow_internal.Size):
            if not isinstance(shape, tuple):
                shape = tuple(shape)
            shape = oneflow._oneflow_internal.Size(shape)
        data_initializer = (
            data_initializer
            if data_initializer is not None
            else flow.empty_initializer(dtype=dtype)
        )
        device = (
            device if device is not None else oneflow._oneflow_internal.device("cpu")
        )
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.placement = placement
        self.sbp = sbp
        self.is_consistent = is_consistent
        self.is_lazy = is_lazy
        self.data_initializer = data_initializer
        self.numpy_data = numpy_data

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


def _default_initializer_for_determining(tensor):
    assert not tensor.is_determined
    undetermined_tensor = tensor._undetermined_tensor
    if undetermined_tensor.is_consistent:
        raise NotImplementedError()
    else:
        shape = undetermined_tensor.shape
        dtype = undetermined_tensor.dtype
        determined_tensor = oneflow._oneflow_internal.LocalTensor(
            shape,
            dtype,
            undetermined_tensor.device,
            undetermined_tensor.is_lazy,
            undetermined_tensor.requires_grad,
            True,
        )
        _init_eager_local_tensor_by_initializer_conf(
            determined_tensor, undetermined_tensor.data_initializer
        )
    return determined_tensor


def _numpy_initializer_for_determining(tensor):
    assert not tensor.is_determined
    undetermined_tensor = tensor._undetermined_tensor
    numpy_data = undetermined_tensor.numpy_data
    assert numpy_data is not None

    if undetermined_tensor.is_consistent:
        raise NotImplementedError()
    else:
        determined_tensor = oneflow._oneflow_internal.LocalTensor(
            undetermined_tensor.shape,
            undetermined_tensor.dtype,
            undetermined_tensor.device,
            undetermined_tensor.is_lazy,
            undetermined_tensor.requires_grad,
            True,
        )
        _copy_from_numpy_to_eager_local_tensor(determined_tensor, numpy_data)

    return determined_tensor


def _input_args_is_tuple_or_list(*args):
    return len(args) == 1 and isinstance(args[0], (tuple, list))


def _input_args_is_numpy(*args):
    return len(args) == 1 and isinstance(args[0], np.ndarray)


def _input_args_is_consistent_or_local(*args):
    return len(args) == 1 and isinstance(
        args[0],
        (
            oneflow._oneflow_internal.ConsistentTensor,
            oneflow._oneflow_internal.LocalTensor,
        ),
    )


def _input_args_is_tensor(*args):
    return len(args) == 1 and isinstance(args[0], flow.Tensor)


def _input_args_is_data(*args):
    return _input_args_is_numpy(*args) or _input_args_is_tuple_or_list(*args)


def _input_args_is_shape(*args):
    return all(isinstance(x, int) for x in args)


def register_tensor_op(op_name):
    def set_tensor_op(method):
        setattr(Tensor, op_name, method)
        setattr(oneflow._oneflow_internal.LocalTensor, op_name, method)
        return method

    return set_tensor_op


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


def _is_scalar(data):
    return isinstance(data, (int, float, bool, complex))


def _flatten_list_or_tuple(list_or_tuple):
    for item in list_or_tuple:
        if isinstance(item, (list, tuple)):
            yield from _flatten_list_or_tuple(item)
        else:
            yield item


def _input_dtype_is_float(data):
    if _is_scalar(data):
        return isinstance(data, float)
    elif isinstance(data, (list, tuple)):
        return any(isinstance(x, float) for x in _flatten_list_or_tuple(data))
    return False
