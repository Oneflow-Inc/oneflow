import numpy as np
import oneflow as flow
from collections import OrderedDict
from test_util import GenArgList


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float
    elif np_dtype == np.int32:
        return flow.int32
    elif np_dtype == np.int64:
        return flow.int64
    else:
        raise NotImplementedError


def _random_input(shape, dtype):
    if dtype == np.float32:
        rand_ = np.random.random_sample(shape).astype(np.float32)
        rand_[np.nonzero(rand_ < 0.5)] = 0.0
        return rand_
    elif dtype == np.int32:
        return np.random.randint(low=0, high=2, size=shape).astype(np.int32)
    else:
        raise NotImplementedError


def _of_argwhere(x, index_dtype, device_type="gpu", dynamic=False):
    data_type = _np_dtype_to_of_dtype(x.dtype)
    out_data_type = _np_dtype_to_of_dtype(index_dtype)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)

    def do_argwhere(x_blob):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.argwhere(x_blob, dtype=out_data_type)

    if dynamic is True:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.function(func_config)
        def argwhere_fn(x_def=flow.MirroredTensorDef(x.shape, dtype=data_type)):
            return do_argwhere(x_def)

        return argwhere_fn([x]).get().ndarray_list()[0]

    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.function(func_config)
        def argwhere_fn(x_def=flow.FixedTensorDef(x.shape, dtype=data_type)):
            return do_argwhere(x_def)

        return argwhere_fn(x).get().ndarray_list()[0]


def _compare_with_np(
    test_case, shape, value_dtype, index_dtype, device_type, dynamic, verbose=False
):
    x = _random_input(shape, value_dtype)
    y = np.argwhere(x)
    of_y = _of_argwhere(x, index_dtype, device_type, dynamic)
    if verbose is True:
        print("input:", x)
        print("np result:", y)
        print("of result:", of_y)
    test_case.assertTrue(np.array_equal(y, of_y))


def test_argwhere(test_case):
    arg_dict = OrderedDict()
    arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
    arg_dict["value_dtype"] = [np.float32, np.int32]
    arg_dict["index_dtype"] = [np.int32, np.int64]
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["dynamic"] = [True, False]
    arg_dict["verbose"] = [False]
    for arg in GenArgList(arg_dict):
        _compare_with_np(test_case, *arg)
