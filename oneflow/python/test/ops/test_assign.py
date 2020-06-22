from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgDict

flow_to_np_dtype_dict = {
    flow.int32: np.int32,
    flow.float: np.single,
    flow.double: np.float,
}


def _random_input(shape, dtype):
    if np.issubdtype(dtype, np.integer):
        return np.random.random_integers(low=-10, high=10, size=shape)
    elif np.issubdtype(dtype, np.floating):
        rng = np.random.default_rng()
        return rng.standard_normal(size=shape, dtype=dtype)
    else:
        raise NotImplementedError


def _of_assign_and_relu(value, dtype, device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(1)
    flow.config.cpu_device_num(1)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)
    func_config.default_placement_scope(flow.fixed_placement(device_type, "0:0"))

    @flow.global_function(func_config)
    def assign_fn(value_def=flow.FixedTensorDef(value.shape, dtype=dtype)):
        var = flow.get_variable(
            name="var",
            shape=value.shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0),
        )
        flow.assign(var, value_def)

    @flow.global_function(func_config)
    def relu_fn():
        var = flow.get_variable(
            name="var",
            shape=value.shape,
            dtype=dtype,
            initializer=flow.constant_initializer(0),
        )
        return flow.nn.relu(var)

    assign_fn(value)
    return relu_fn().get().ndarray()


def _np_relu(x):
    return np.maximum(x, 0)


def _compare_with_np(test_case, shape, dtype, device_type):
    x = _random_input(shape, flow_to_np_dtype_dict[dtype])
    of_y = _of_assign_and_relu(x, dtype, device_type)
    test_case.assertTrue(np.allclose(_np_relu(x), of_y))


def test_assign(test_case):
    arg_dict = OrderedDict()
    arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
    arg_dict["dtype"] = [flow.float, flow.double]
    arg_dict["device_type"] = ["cpu", "gpu"]
    for arg in GenArgDict(arg_dict):
        _compare_with_np(test_case, **arg)
