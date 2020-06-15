import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import Args, CompareOpWithTensorFlow, GenArgDict

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def test_reshape(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["flow_op"] = [flow.reshape]
    arg_dict["tf_op"] = [tf.reshape]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict["op_args"] = [Args([(100, 10)]), Args([(10, 100)]), Args([(5, 20, 10)])]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)
