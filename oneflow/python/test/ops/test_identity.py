import numpy as np
import oneflow as flow
from collections import OrderedDict
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type
from test_util import GenArgList

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _run_test(test_case, device_type, shape, data_type):
    flow.clear_default_session()

    @flow.function(func_config)
    def test_job(x=flow.FixedTensorDef(shape, dtype=type_name_to_flow_type[data_type])):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.identity(x)

    np_type = type_name_to_np_type[data_type]
    if np.issubdtype(np_type, np.integer):
        x_arr = np.random.randint(low=-128, high=127, size=shape).astype(np_type)
    else:
        x_arr = np.random.rand(*shape).astype(np_type)
    y_arr = test_job(x_arr).get().ndarray()
    test_case.assertTrue(np.array_equal(x_arr, y_arr))


def test_identity(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["shape"] = [(96, 96)]
    arg_dict["data_type"] = ["int8", "int32", "int64", "float32", "double"]
    for arg in GenArgList(arg_dict):
        _run_test(test_case, *arg)
