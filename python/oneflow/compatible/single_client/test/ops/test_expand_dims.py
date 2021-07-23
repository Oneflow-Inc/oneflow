import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from test_util import GenArgList

from oneflow.compatible import single_client as flow

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, x_shape, axis):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def check_grad(x_diff_blob):
        assert np.array_equal(x_diff_blob.numpy(), np.ones(x_shape))

    @flow.global_function(type="train", function_config=func_config)
    def ExpandDimsJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "var",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.ones_initializer(),
                trainable=True,
            )
            flow.watch_diff(x, check_grad)
            loss = flow.expand_dims(x, axis)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
            ).minimize(loss)
            return loss

    of_out = ExpandDimsJob().get().numpy()
    tf_out = tf.expand_dims(np.ones(x_shape, dtype=np.float32), axis).numpy()
    assert np.array_equal(of_out, tf_out)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(10, 10)]
    arg_dict["axis"] = [0, 1, 2, -1, -2, -3]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestExpandDims(flow.unittest.TestCase):
    def test_expand_dims(test_case):
        for arg in gen_arg_list():
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
