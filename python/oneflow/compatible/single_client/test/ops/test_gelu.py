import math
import os
import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from test_util import GenArgDict, RunOneflowOp

from oneflow.compatible import single_client as flow

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def tf_gelu(x):
    inv_sqrt2 = math.sqrt(0.5)
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = 0.5 * x * (1 + tf.math.erf(inv_sqrt2 * x))
    x_diff = tape.gradient(y, x)
    return (y.numpy(), x_diff.numpy())


@flow.unittest.skip_unless_1n1d()
class TestGelu(flow.unittest.TestCase):
    def test_gelu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["flow_op"] = [flow.math.gelu]
        arg_dict["flow_args"] = [[]]
        arg_dict["x"] = [
            np.random.uniform(low=-100, high=100, size=(10, 20, 30, 40)).astype(
                np.float32
            )
        ]
        for arg in GenArgDict(arg_dict):
            (of_y, of_x_diff) = RunOneflowOp(**arg)
            (tf_y, tf_x_diff) = tf_gelu(arg["x"])
            assert np.allclose(of_y, tf_y, rtol=1e-05, atol=1e-05)
            assert np.allclose(of_x_diff, tf_x_diff, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    unittest.main()
