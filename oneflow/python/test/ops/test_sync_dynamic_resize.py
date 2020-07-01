from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def test_sync_dynamic_resize(_):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [
        (100,),
        (100, 1),
        (1000, 10),
        (10, 10, 200),
    ]
    arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
    arg_dict["size_type"] = ["int32", "int64"]

    for device_type, x_shape, data_type, size_type in GenArgList(arg_dict):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)

        @flow.global_function(func_config)
        def TestJob(
            x=flow.FixedTensorDef(x_shape, dtype=type_name_to_flow_type[data_type]),
            size=flow.FixedTensorDef((1,), dtype=type_name_to_flow_type[size_type]),
        ):
            with flow.fixed_placement(device_type, "0:0"):
                return flow.sync_dynamic_resize(x, size)

        size = np.random.randint(0, x_shape[0])
        x = np.random.rand(*x_shape).astype(type_name_to_np_type[data_type])
        y = (
            TestJob(x, np.array([size]).astype(type_name_to_np_type[size_type]))
            .get()
            .ndarray_list()[0]
        )
        assert np.array_equal(y, x[:size])
