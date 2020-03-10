import oneflow as flow
import numpy as np
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def test1(_):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["x_shape"] = [
        (100,),
        (100, 100),
        (1000, 1000),
        (10, 10, 2000),
        (10, 10000),
    ]
    arg_dict["data_type"] = ["float32", "double", "int32", "int64"]

    for device_type, x_shape, data_type in GenArgList(arg_dict):
        assert device_type in ["gpu", "cpu"]
        assert data_type in ["float32", "double", "int8", "int32", "int64"]
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)

        @flow.function(flow.FunctionConfig())
        def TestJob(
            x=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[data_type]),
            size=flow.FixedTensorDef((1,), dtype=flow.int32),
        ):
            with flow.fixed_placement(device_type, "0:0"):
                return flow.sync_dynamic_resize(x, size)

        size = np.random.randint(0, x_shape[0])
        x = np.random.rand(*x_shape).astype(type_name_to_np_type[data_type])
        y = TestJob(x, np.array([size]).astype(np.int32)).get().ndarray_list()[0]
        assert np.array_equal(y, x[:size])
