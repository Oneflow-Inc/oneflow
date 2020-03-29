import oneflow as flow
import numpy as np
from collections import OrderedDict
import uuid
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
import random

def gen_numpy_data(x, label, beta=1.0, scale=1.0):
    original_shape = x.shape
    elem_cnt = x.size
    x = x.reshape(-1)
    label = label.reshape(-1)
    y = np.zeros((elem_cnt)).astype(np.float)
    dx = np.zeros((elem_cnt)).astype(np.float)

    # Forward
    for i in np.arange(elem_cnt):
        abs_diff = abs(x[i] - label[i])
        if abs_diff < beta:
            y[i] = 0.5 * abs_diff * abs_diff / beta
        else:
            y[i] = abs_diff - 0.5 * beta
    y *= scale

    # Backward
    for i in np.arange(elem_cnt):
        diff = x[i] - label[i]
        abs_diff = abs(diff)
        if abs_diff < beta:
            dx[i] = diff / beta
        else:
            dx[i] = np.sign(diff)
    
    dx *= scale
    return {
        "y": y.reshape(original_shape),
        "dx": dx.reshape(original_shape)
    }

def test_smooth_l1(_):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [
        (100,),
        (10, 10),
    ]
    arg_dict["data_type"] = ["float32", "double"]
    arg_dict["beta"] = [0, 0.5, 1]
    arg_dict["scale"] = [-1.1, 0, 1]

    for case in GenArgList(arg_dict):
        device_type, x_shape, data_type, beta, scale = case
        assert device_type in ["gpu", "cpu"]
        assert data_type in ["float32", "double", "int8", "int32", "int64"]
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.train.primary_lr(1e-4)
        func_config.train.model_update_conf(dict(naive_conf={}))

        x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
        label = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])

        np_result = gen_numpy_data(x, label, beta, scale)

        def assert_dx(b):
            dx_np = np_result["dx"].astype(type_name_to_np_type[data_type])
            assert np.allclose(dx_np, b.ndarray()), (case, dx_np, b.ndarray())

        @flow.function(func_config)
        def TestJob(
            x=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[data_type]),
            label=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[data_type])
        ):
            v = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.constant_initializer(0),
                trainable=True,
            )
            flow.watch_diff(v, assert_dx)
            x += v
            with flow.fixed_placement(device_type, "0:0"):
                y = flow.smooth_l1(x, label, beta, scale)
                flow.losses.add_loss(y)
                return y
        
        y_np = np_result["y"].astype(type_name_to_np_type[data_type])
        y = TestJob(x, label).get().ndarray()
        assert np.allclose(y_np, y), (case, y_np, y)
