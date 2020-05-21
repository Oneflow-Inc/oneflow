import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict 

from test_util import GenArgDict
from test_util import CompareOpWithTensorFlow
from test_util import Args



def test_pad_gpu(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict['flow_op'] = [flow.pad]
    arg_dict['tf_op'] = [tf.pad]
    arg_dict["input_shape"] = [(2, 2, 1, 3), (1, 1, 2, 3)]
    arg_dict["op_args"] = [
                            Args([([0, 0], [0, 0], [1, 2], [1, 1])], tf.constant([([0, 0], [0, 0], [1, 2], [1, 1])])), 
                            Args([([0, 0], [0, 0], [0, 1], [1, 0])], tf.constant([([0, 0], [0, 0], [0, 1], [1, 0])])), 
                            Args([([0, 0], [0, 0], [10, 20], [0, 0])], tf.constant([([0, 0], [0, 0], [10, 20], [0, 0])])), 
                           ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)

def test_pad_cpu(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu"]
    arg_dict['flow_op'] = [flow.pad]
    arg_dict['tf_op'] = [tf.pad]
    arg_dict["input_shape"] = [(2, 3, 4, 3), (5, 1, 1, 1)]
    arg_dict["op_args"] = [
                            Args([([0, 0], [0, 0], [1, 2], [1, 1])], tf.constant([([0, 0], [0, 0], [1, 2], [1, 1])])), 
                            Args([([0, 0], [0, 0], [0, 1], [1, 0])], tf.constant([([0, 0], [0, 0], [0, 1], [1, 0])])), 
                            Args([([0, 0], [0, 0], [10, 20], [0, 0])], tf.constant([([0, 0], [0, 0], [10, 20], [0, 0])])), 
                           ]
    for arg in GenArgDict(arg_dict):
        CompareOpWithTensorFlow(**arg)
