import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict 

from test_util import GenArgDict
from test_util import CompareOpWithTensorFlow
from test_util import Args


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def test_scalar_mul(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict['flow_op'] = [flow.math.multiply]
    arg_dict['tf_op'] = [tf.math.multiply]
    arg_dict["input_shape"] = [(10, 10, 10)]
    arg_dict['op_args'] = [Args([1]), Args([-1]), Args([84223.19348]), Args([-3284.139])] 
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)

