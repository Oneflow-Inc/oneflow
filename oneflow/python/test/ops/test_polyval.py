import numpy as np
import tensorflow as tf
import oneflow as flow
import oneflow.typing as tp
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, in_shape, data_type):
    assert device_type in ["gpu", "cpu"]
    assert data_type in ["float32", "double", "int8", "int32", "int64"]
    flow_data_type = type_name_to_flow_type[data_type]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow_data_type)

    @flow.global_function(function_config=func_config)
    def PolyValJob0(
        x: tp.Numpy.Placeholder(in_shape, dtype=flow_data_type)
    ):
        return flow.math.polyval([], x)

    @flow.global_function(function_config=func_config)
    def PolyValJob3(
        x: tp.Numpy.Placeholder(in_shape, dtype=flow_data_type),
        coeffs0: tp.Numpy.Placeholder((1,), dtype=flow_data_type),
        coeffs1: tp.Numpy.Placeholder((1,), dtype=flow_data_type),
        coeffs2: tp.Numpy.Placeholder((1,), dtype=flow_data_type)
    ):
        return flow.math.polyval([coeffs0, coeffs1, coeffs2], x)

    x = (np.random.random(in_shape) *
         100).astype(type_name_to_np_type[data_type])

    coeffs = [(np.random.random(1)*100).astype(type_name_to_np_type[data_type])
              for i in range(3)]

    # OneFlow
    of_out = []
    of_out.append(PolyValJob0(x).get().numpy())
    of_out.append(PolyValJob3(
        x, coeffs[0], coeffs[1], coeffs[2]).get().numpy())
    # TensorFlow
    tf_out = []
    tf_out.append(tf.math.polyval([], x).numpy())
    tf_out.append(tf.math.polyval(coeffs, x).numpy())
    for i in range(2):
        assert np.allclose(of_out[i], tf_out[i], rtol=1e-5, atol=1e-5)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(1,), (2, 1), (1, 2), (2, 2)]
    arg_dict["data_type"] = ["float32", "double"]
    return GenArgList(arg_dict)


def test_polyval(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
