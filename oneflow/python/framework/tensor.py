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
import oneflow_api
import numpy as np
import inspect
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.check_point_v2 as check_point_v2
from oneflow.python.framework.function_util import global_function_or_identity
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow as flow
from oneflow.python.nn.modules import *


@oneflow_export("Tensor")
class Tensor:
    def __init__(
        self,
        *args,
        dtype=None,
        device=None,
        requires_grad=False,
        retain_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
        data_initializer=None,
        determining_initializer=None,
    ):
        assert len(args) > 0
        dtype = dtype if dtype is not None else oneflow_api.float32
        if placement is None:
            device = device if device is not None else oneflow_api.device("cpu")
        if _input_args_is_tensor(*args):
            TODO()  # liyurui, construct using another tensor
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
                retain_grad=retain_grad,
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

    def size(self):
        return self.shape

    def dim(self, idx):
        return self.shape[idx]

    def ndimension(self):
        return self.ndim

    @_auto_determine
    def detach(self):
        if self._local_or_consistent_tensor is not None:
            return flow.Tensor(self._local_or_consistent_tensor.detach())
        else:
            return None

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

    @_auto_determine
    def numpy(self):
        return remote_blob_util.BlobObjectNumpy(
            self._local_or_consistent_tensor._blob_object
        )

    def tolist(self):
        TODO()

    def backward(
        self, gradient=None, retain_graph=False, create_graph=False, inputs=None
    ):
        assert self.is_determined
        TODO()  # liyurui

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "[Tensor shape={} dtype={}]".format(self.shape, self.dtype)

    def __array__(self):
        TODO()

    def __sizeof__(self):
        TODO()

    def __deepcopy__(self, memo):
        TODO()

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

    def uniform_(self, a=0, b=1):
        initializer_conf = flow.random_uniform_initializer(
            minval=a, maxval=b, dtype=self.dtype
        )
        return self._init_by_initializer_conf(initializer_conf)

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

    def xavier_normal_(self, gain=1.0, *, data_format="NCHW"):
        assert gain == 1.0, "Only gain == 1.0 is supported now"
        initializer_conf = flow.xavier_normal_initializer(data_format=data_format)
        return self._init_by_initializer_conf(initializer_conf)

    def xavier_uniform_(self, gain=1.0, *, data_format="NCHW"):
        assert gain == 1.0, "Only gain == 1.0 is supported now"
        initializer_conf = flow.xavier_uniform_initializer(data_format=data_format)
        return self._init_by_initializer_conf(initializer_conf)

    def normal_(self, mean=0, std=1):
        initializer_conf = flow.random_normal_initializer(mean=mean, stddev=std)
        return self._init_by_initializer_conf(initializer_conf)

    def fill_(self, value):
        initializer_conf = flow.constant_initializer(value=value, dtype=self.dtype)
        return self._init_by_initializer_conf(initializer_conf)

    def _init_by_initializer_conf(self, initializer_conf):
        if self.is_determined:
            with self._placement_scope():
                check_point_v2.init_by_initializer_conf(
                    self, initializer_conf, True, None
                )
        else:
            self.set_data_initializer(initializer_conf)
        return self

    def _placement_scope(self):
        if self.is_consistent:
            return _convert_to_placement_scope(self.placement)
        else:
            return _convert_to_placement_scope(self.device)

    @property
    @_auto_determine
    def _blob_object(self):
        return self._local_or_consistent_tensor._blob_object

    def _construct_with_data(
        self,
        *args,
        dtype=None,
        device=None,
        requires_grad=False,
        retain_grad=False,
        placement=None,
        sbp=None,
        is_consistent=False,
        is_lazy=False,
    ):
        numpy_data = None
        if _input_args_is_tuple_or_list(*args):
            numpy_data = np.array(args[0]).astype(
                flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
            )
        elif _input_args_is_numpy(*args):
            numpy_data = args[0].astype(
                flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
            )
        shape = oneflow_api.Size(tuple(numpy_data.shape))
        self._determining_initializer = _numpy_initializer_for_determining
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
            numpy_data=numpy_data,
        )


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
        numpy_data=None,
    ):
        if not isinstance(shape, oneflow_api.Size):
            if not isinstance(shape, tuple):
                shape = tuple(shape)
            shape = oneflow_api.Size(shape)
        data_initializer = (
            data_initializer
            if data_initializer is not None
            else flow.empty_initializer(dtype=dtype)
        )
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
    variable_name = id_util.UniqueStr("tensor_")

    blob = None

    @global_function_or_identity()
    def job():
        nonlocal blob
        with tensor._placement_scope():
            blob = flow.get_variable(
                name=variable_name,
                shape=tuple(undetermined_tensor.shape),
                dtype=undetermined_tensor.dtype,
                initializer=undetermined_tensor.data_initializer,
            )

    job()
    if undetermined_tensor.is_consistent:
        determined_tensor = oneflow_api.ConsistentTensor(
            undetermined_tensor.shape,
            undetermined_tensor.dtype,
            undetermined_tensor.sbp,
            undetermined_tensor.placement,
            undetermined_tensor.is_lazy,
            undetermined_tensor.requires_grad,
            True,
            undetermined_tensor.retain_grad,
        )
    else:
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
    return determined_tensor


def _numpy_initializer_for_determining(tensor):
    assert not tensor.is_determined
    undetermined_tensor = tensor._undetermined_tensor
    assert undetermined_tensor.numpy_data is not None
    variable_name = id_util.UniqueStr("tensor_")

    @global_function_or_identity()
    def set_numpy_data():
        with tensor._placement_scope():
            flow.get_variable(
                name=variable_name,
                shape=tuple(undetermined_tensor.shape),
                dtype=undetermined_tensor.dtype,
                initializer=undetermined_tensor.data_initializer,
            )

    set_numpy_data()
    flow.load_variables({variable_name: undetermined_tensor.numpy_data})
    blob = flow.get_all_variables()[variable_name]
    if undetermined_tensor.is_consistent:
        determined_tensor = oneflow_api.ConsistentTensor(
            undetermined_tensor.shape,
            undetermined_tensor.dtype,
            undetermined_tensor.sbp,
            undetermined_tensor.placement,
            undetermined_tensor.is_lazy,
            undetermined_tensor.requires_grad,
            True,
            undetermined_tensor.retain_grad,
        )
    else:
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
    return determined_tensor


def _input_args_is_tuple_or_list(*args):
    return len(args) == 1 and isinstance(args[0], (tuple, list))


def _input_args_is_numpy(*args):
    return len(args) == 1 and isinstance(args[0], np.ndarray)


def _input_args_is_consistent_or_local(*args):
    return len(args) == 1 and isinstance(
        args[0], (oneflow_api.ConsistentTensor, oneflow_api.LocalTensor)
    )


def _input_args_is_tensor(*args):
    return len(args) == 1 and isinstance(args[0], flow.Tensor)


def _input_args_is_data(*args):
    return _input_args_is_numpy(*args) or _input_args_is_tuple_or_list(*args)


def _input_args_is_shape(*args):
    return all(isinstance(x, int) for x in args)


def register_tensor_op_by_module(op_name):
    def set_method(module):
        if is_unary_module(module):
            setattr(
                Tensor,
                op_name,
                lambda self, *args, **kwargs: module(*args, **kwargs).forward(self),
            )
        else:
            assert is_binary_module(module)
            setattr(
                Tensor,
                op_name,
                lambda self, x, *args, **kwargs: module(*args, **kwargs).forward(
                    self, x
                ),
            )
        return module

    return set_method


def register_op_by_module(op_name):
    def set_method(module):
        if is_unary_module(module):
            oneflow_export(op_name)(_get_unary_module_impl(module))
        else:
            assert is_binary_module(module)
            oneflow_export(op_name)(_get_binary_module_impl(module))

        return module

    def _get_unary_module_impl(module):
        def unary_module_impl(x, *args, **kwargs):
            return module(*args, **kwargs).forward(x)

        name = module.__name__ + "_op"
        unary_module_impl.__name__ = name
        globals()[name] = unary_module_impl

        return unary_module_impl

    def _get_binary_module_impl(module):
        def binary_module_impl(x, y, *args, **kwargs):
            return module(*args, **kwargs).forward(x, y)

        name = module.__name__ + "_op"
        binary_module_impl.__name__ = name
        globals()[name] = binary_module_impl

        return binary_module_impl

    return set_method


def is_unary_module(module):
    return True if len(inspect.signature(module.forward).parameters) == 2 else False


def is_binary_module(module):
    return True if len(inspect.signature(module.forward).parameters) == 3 else False


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
