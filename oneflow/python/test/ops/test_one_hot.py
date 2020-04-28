import oneflow as flow
import numpy as np
from collections import OrderedDict 
from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def _check(test_case, x, y, depth):
    np_one_hot = np.eye(depth)[x]
    test_case.assertTrue(np.allclose(np_one_hot, y))

def _run_test(test_case, device_type, x_shape, depth, dtype, out_dtype):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    @flow.function(func_config)
    def one_hotJob(x=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[dtype])):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.one_hot(x, depth=depth, dtype=type_name_to_flow_type[out_dtype])
    x = np.random.randint(0, depth, x_shape).astype(type_name_to_np_type[dtype])
    y = one_hotJob(x).get()
    _check(test_case, x, y.ndarray(), depth)
   
def test_one_hot(test_case):
    arg_dict = OrderedDict()
    arg_dict["test_case"] = [test_case]
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [(10, 20, 30)]
    arg_dict["depth"] = [10, 20]
    arg_dict["dtype"] = ["int32", "int64"]
    arg_dict["out_dtype"] = ["int32", "int64", "float32", "double"]
    for arg in GenArgList(arg_dict):
        _run_test(*arg)
