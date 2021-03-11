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
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.runtime_mode as rt_mode
import oneflow as flow

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
        determining_initializer=None
    ):
        assert len(args) > 0
        dtype = dtype if dtype is not None else oneflow_api.float32
        device = device if device is not None else oneflow_api.device("cpu")
        if _input_args_is_other_data(*args):
            self._immediately_construct(
                *args,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                retain_grad=retain_grad,
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

    @property
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

    # internal decorator
    def _auto_determine(func):
        def wrapped_func(*args, **kwargs):
            tensor = args[0]
            if not tensor.is_determined:
                tensor.determine()
            return func(*args, **kwargs)

        return wrapped_func

    def retain_grad(self):
        assert self.is_determined
        self._local_or_consistent_tensor.retain_grad()

    def data_ptr(self):
        TODO()

    def element_size(self):
        return self.dtype.bytes

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

    def backward(
        self, gradient=None, retain_graph=False, create_graph=False, inputs=None
    ):
        assert self.is_determined
        TODO()  # liyurui

    def __str__(self):
        TODO()

    def __repr__(self):
        return "[Tensor shape={} dtype={}]".format(self.shape, self.dtype)

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

    def _construct_determined_tensor_with_numpy(
        self,
        dtype=None,
        device=None,
        requires_grad=False,
        retain_grad=False,
        is_lazy=False,
        numpy_data=None,
    ):
        shape = oneflow_api.Size(tuple(numpy_data.shape))
        # Only local tensor will be created
        self._local_or_consistent_tensor = _initialized_job(
            shape=shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            retain_grad=retain_grad,
            is_lazy=is_lazy,
            numpy_data=numpy_data,
        )
        self._undetermined_tensor = None

    def _immediately_construct(
        self,
        *args,
        dtype=None,
        device=None,
        requires_grad=False,
        retain_grad=False,
        is_lazy=False
    ):
        if _input_args_is_tuple_or_list(*args):
            numpy_data = np.array(args[0]).astype(
                flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
            )
            self._construct_determined_tensor_with_numpy(
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                retain_grad=retain_grad,
                is_lazy=is_lazy,
                numpy_data=numpy_data,
            )
        elif _input_args_is_numpy(*args):
            numpy_data = args[0].astype(
                flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
            )
            self._construct_determined_tensor_with_numpy(
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                retain_grad=retain_grad,
                is_lazy=is_lazy,
                numpy_data=numpy_data,
            )
        elif _input_args_is_consistent_or_mirrored(*args):
            self._local_or_consistent_tensor = args[0]
            self._undetermined_tensor = None


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
    assert not undetermined_tensor.is_consistent
    blob = flow.get_variable(
        name=id_util.UniqueStr("tensor_"),
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
    return determined_tensor


def global_function_or_identity(*args, **kwargs):
    if rt_mode.CurrentMode() == rt_mode.NORMAL_MODE:
        return flow.global_function(*args, **kwargs)
    else:
        assert rt_mode.CurrentMode() == rt_mode.GLOBAL_MODE
        identity_decorator = lambda func: func
        return identity_decorator

def _initialized_job(
    shape=None,
    dtype=None,
    device=None,
    requires_grad=None,
    retain_grad=None,
    is_lazy=False,
    numpy_data=None,
):
    assert numpy_data is not None
    variable_name = id_util.UniqueStr("tensor_")

    @global_function_or_identity()
    def set_data():
        flow.get_variable(
            name=variable_name,
            shape=tuple(shape),
            dtype=dtype,
            initializer=flow.zeros_initializer(dtype=dtype),
        )

    if not is_lazy:
        set_data()
    flow.load_variables({variable_name: numpy_data})
    blob = flow.get_all_variables()[variable_name]
    determined_tensor = oneflow_api.LocalTensor(
        shape, dtype, device, is_lazy, requires_grad, True, retain_grad,
    )
    determined_tensor._set_blob_object(blob.blob_object)
    return determined_tensor


def _input_args_is_tuple_or_list(*args):
    return len(args) == 1 and isinstance(args[0], (tuple, list))


def _input_args_is_numpy(*args):
    return len(args) == 1 and isinstance(args[0], np.ndarray)


def _input_args_is_consistent_or_mirrored(*args):
    return len(args) == 1 and isinstance(
        args[0], (oneflow_api.ConsistentTensor, oneflow_api.LocalTensor)
    )


def _input_args_is_other_data(*args):
    return (
        _input_args_is_consistent_or_mirrored(*args)
        or _input_args_is_numpy(*args)
        or _input_args_is_tuple_or_list(*args)
    )


def _input_args_is_shape(*args):
    return all(isinstance(x, int) for x in args)
