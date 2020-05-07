import numpy as np
import tensorflow as tf
import oneflow as flow
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
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(type_name_to_flow_type[data_type])

    @flow.function(func_config)
    def PolyValJob0(x=flow.FixedTensorDef(in_shape)):
        return flow.math.polyval([],x)
    
    @flow.function(func_config)
    def PolyValJob1(x=flow.FixedTensorDef(in_shape), coeffs0=flow.FixedTensorDef((1,))):
        return flow.math.polyval([coeffs0], x)
    
    @flow.function(func_config)
    def PolyValJob2(x=flow.FixedTensorDef(in_shape), coeffs0=flow.FixedTensorDef((1,)),coeffs1=flow.FixedTensorDef((1,)), coeffs2=flow.FixedTensorDef((1,))):
        return flow.math.polyval([coeffs0, coeffs1, coeffs2],x)

    @flow.function(func_config)
    def PolyValJob3(x=flow.FixedTensorDef(in_shape), coeffs0=flow.FixedTensorDef((1,)),coeffs1=flow.FixedTensorDef((1,)), coeffs2=flow.FixedTensorDef((1,)),
                    coeffs3=flow.FixedTensorDef((1,)), coeffs4=flow.FixedTensorDef((1,))):
        return flow.math.polyval([coeffs0,coeffs1,coeffs2,coeffs3,coeffs4],x)

    x = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])
    coeffs_len = 5
    coeffs=[]
    for i in range(coeffs_len):
        coeffs.append((np.random.random(1)*100).astype(type_name_to_np_type[data_type]))
    
    # OneFlow
    of_out=[]
    of_out.append(PolyValJob0(x).get().ndarray())
    of_out.append(PolyValJob1(x, coeffs[0]).get().ndarray())
    of_out.append(PolyValJob2(x, coeffs[0], coeffs[1], coeffs[2]).get().ndarray())
    of_out.append(PolyValJob3(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]).get().ndarray())
    # TensorFlow
    tf_out = []
    tf_out.append(tf.math.polyval([], x).numpy())
    tf_out.append(tf.math.polyval([coeffs[0]], x).numpy())
    tf_out.append(tf.math.polyval([coeffs[0], coeffs[1], coeffs[2]], x).numpy())
    tf_out.append(tf.math.polyval([coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]], x).numpy())
    for i in range(4):
        assert np.allclose(of_out[i], tf_out[i], rtol=1e-5, atol=1e-5)


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["in_shape"] = [(1,), (5,), (5, 5), (5, 5, 5)]
    arg_dict["data_type"] = ["float32"]
    return GenArgList(arg_dict)


def test_polyval(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)
