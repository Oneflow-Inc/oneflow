from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import (
    Args,
    CompareOpWithTensorFlow,
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def test_scalar_mul(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["flow_op"] = [flow.math.multiply]
    arg_dict["tf_op"] = [tf.math.multiply]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict["op_args"] = [
        Args([1]),
        Args([-1]),
        Args([84223.19348]),
        Args([-3284.139]),
    ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)


def _test_element_wise_mul_fw_bw(test_case, device, shape, type_name):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    np_type = type_name_to_np_type[type_name]
    flow_type = type_name_to_flow_type[type_name]

    @flow.global_function(func_config)
    def test_element_wise_mul_job(
        x=flow.FixedTensorDef(shape, dtype=flow_type),
        y=flow.FixedTensorDef(shape, dtype=flow_type),
    ):
        with flow.fixed_placement(device, "0:0"):
            x += flow.get_variable(
                name="vx",
                shape=(1,),
                dtype=flow_type,
                initializer=flow.zeros_initializer(),
            )
            y += flow.get_variable(
                name="vy",
                shape=(1,),
                dtype=flow_type,
                initializer=flow.zeros_initializer(),
            )
            out = flow.math.multiply(x, y)
            flow.losses.add_loss(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(y, test_global_storage.Setter("y"))
            flow.watch_diff(y, test_global_storage.Setter("y_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    check_point = flow.train.CheckPoint()
    check_point.init()
    test_element_wise_mul_job(
        np.random.rand(*shape).astype(np_type), np.random.rand(*shape).astype(np_type)
    ).get()
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("x") * test_global_storage.Get("y"),
            test_global_storage.Get("out"),
        )
    )
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("out_diff") * test_global_storage.Get("x"),
            test_global_storage.Get("y_diff"),
        )
    )
    test_case.assertTrue(
        np.allclose(
            test_global_storage.Get("out_diff") * test_global_storage.Get("y"),
            test_global_storage.Get("x_diff"),
        )
    )


def test_element_wise_mul_fw_bw(test_case):
    arg_dict = OrderedDict()
    arg_dict["device"] = ["gpu", "cpu"]
    arg_dict["shape"] = [(96, 96)]
    arg_dict["type_name"] = ["float32", "double"]
    for arg in GenArgDict(arg_dict):
        _test_element_wise_mul_fw_bw(test_case, **arg)
