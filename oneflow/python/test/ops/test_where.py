import numpy as np
import oneflow as flow
# import tensorflow as tf
from collections import OrderedDict
from test_util import GenArgList


def _random_input(cond_shape, x_shape, y_shape):
    condition = np.random.randint(low=0, high=1, size=cond_shape).astype(np.int32)
    x = np.random.standard_normal(x_shape).astype(np.float32)
    y = np.random.standard_normal(y_shape).astype(np.float32)
    return condition, x, y


def _of_where(condition, x, y, device_type="gpu", dynamic=False, grad_cb=None):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    if callable(grad_cb):
        func_config.train.primary_lr(1e-3)
        func_config.train.model_update_conf(dict(naive_conf={}))
        raise NotImplementedError
    else:

        def do_where(condition, x, y):
            with flow.device_prior_placement(device_type, "0:0"):
                return flow.where(condition, x, y)

    if dynamic:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.function(func_config)
        def where_fn(
            condition_def=flow.MirroredTensorDef(condition.shape, dtype=flow.int32),
            x_def=flow.MirroredTensorDef(x.shape, dtype=flow.float),
            y_def=flow.MirroredTensorDef(y.shape, dtype=flow.float),
        ):
            return do_where(condition_def, x_def, y_def)

        if callable(grad_cb):
            check_point = flow.train.CheckPoint()
            check_point.init()
        return where_fn([condition], [x], [y]).get().ndarray_list()[0]

    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.function(func_config)
        def where_fn(
            condition_def=flow.FixedTensorDef(condition.shape, dtype=flow.int32),
            x_def=flow.FixedTensorDef(x.shape, dtype=flow.float),
            y_def=flow.FixedTensorDef(y.shape, dtype=flow.float),
        ):
            return do_where(condition_def, x_def, y_def)

        if callable(grad_cb):
            check_point = flow.train.CheckPoint()
            check_point.init()
        return where_fn(condition, x, y).get().ndarray()


def _compare_with_np(test_case, cond_shape, x_shape, y_shape, device_type, dynamic):
    condition, x, y = _random_input(cond_shape, x_shape, y_shape)
    z = np.where(condition, x, y)
    of_z = _of_where(condition, x, y, device_type, dynamic)
    test_case.assertTrue(np.array_equal(z, of_z))


def test_where(test_case):
    arg_dict = OrderedDict()
    arg_dict["cond_shape"] = [[4, 5, 8]]
    arg_dict["x_shape"] = [[1, 5, 8]]
    arg_dict["y_shape"] = [[4, 1, 8]]
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgList(arg_dict):
        _compare_with_np(test_case, *arg)
