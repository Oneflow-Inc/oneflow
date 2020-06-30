import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList


def _gen_test_data(params_shape, indices_shape, axis):
    params = np.random.rand(*params_shape).astype(np.float32)
    indices = np.random.randint(
        low=0, high=params_shape[axis], size=indices_shape
    ).astype(np.int32)
    slices = [slice(None)] * len(params_shape)
    slices[axis] = indices
    out = params[tuple(slices)]
    return params, indices, out


def _test_gather_model_parallel_fw(
    test_case, device_type, params_shape, indices_shape, axis, split_axis
):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.global_function(func_config)
    def gather_model_parallel_fw_job(
        params=flow.FixedTensorDef(params_shape, dtype=flow.float),
        indices=flow.FixedTensorDef(indices_shape, dtype=flow.int32),
    ):
        with flow.fixed_placement(device_type, "0:0-3"):
            params = params.with_distribute(flow.distribute.split(split_axis))
            indices = indices.with_distribute(flow.distribute.broadcast())
            return flow.gather(params=params, indices=indices, axis=axis)

    params_arr, indices_arr, out_arr = _gen_test_data(params_shape, indices_shape, axis)
    out = gather_model_parallel_fw_job(params_arr, indices_arr).get().ndarray()
    if axis == split_axis:
        test_case.assertTrue(np.allclose(out, out_arr))
    else:
        test_case.assertTrue(np.array_equal(out, out_arr))


def test_gather_model_parallel_fw(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["params_shape"] = [(96, 96, 96)]
    arg_dict["indices_shape"] = [(32, 48)]
    arg_dict["axis"] = [0, 1, 2]
    arg_dict["split_axis"] = [0, 1, 2]
    for arg in GenArgList(arg_dict):
        if arg[3] == arg[4] and os.getenv("ENABLE_USER_OP") != "True":
            continue
        _test_gather_model_parallel_fw(test_case, *arg)
