import os
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type


def of_run(device_type, x_shape, rate, seed):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    @flow.global_function(func_config)
    def RandomMaskLikeJob(x=flow.FixedTensorDef(x_shape)):
        with flow.device_prior_placement(device_type, "0:0"):
            mask = flow.nn.random_mask_like(x, rate=rate, seed=seed)
            return mask

    # OneFlow
    x = np.random.rand(*x_shape).astype(np.float32)
    of_out = RandomMaskLikeJob(x).get().ndarray()
    assert np.allclose(
        [1 - np.count_nonzero(of_out) / of_out.size], [rate], atol=rate / 5
    )


def test_random_mask_like(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(100, 100, 10, 20), (100, 100, 200)]
    arg_dict["rate"] = [0.1, 0.4, 0.75]
    arg_dict["seed"] = [12345, None]

    for arg in GenArgList(arg_dict):
        of_run(*arg)
