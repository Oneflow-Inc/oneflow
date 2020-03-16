import numpy as np
import oneflow as flow


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float
    elif np_dtype == np.int32:
        return flow.int32
    else:
        raise NotImplementedError


def _random_input(shape, dtype):
    if dtype == np.float32:
        return np.random.standard_normal(shape).astype(np.float32)
    elif dtype == np.int32:
        return np.random.randint(low=0, high=10, size=shape).astype(np.int32)
    else:
        raise NotImplementedError


def _of_argwhere(x, device_type="gpu", dynamic=False):
    data_type = _np_dtype_to_of_dtype(x.dtype)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)

    def do_argwhere(x_blob):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.argwhere(x_blob, index_data_type=flow.int32)

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

        return argwhere_fn(x).get().ndarray()


def test_argwhere(test_case):
    x = _random_input((30, 4), np.float32)
    of_y = _of_argwhere(x, dynamic=False)
    test_case(np.argwhere(x), of_y)
