import numpy as np
import oneflow as flow
import tensorflow as tf
from collections import OrderedDict
from test_util import GenArgList

tf.compat.v1.enable_eager_execution()


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float
    else:
        raise NotImplementedError


def _of_clip_by_value(values, min=None, max=None, device_type="gpu", dynamic=False, grad_cb=None):
    data_type = _np_dtype_to_of_dtype(values.dtype)

    if callable(grad_cb):

        def clip(values_blob):
            with flow.device_prior_placement(device_type, "0:0"):
                x = flow.get_variable(
                    "values",
                    shape=values.shape,
                    dtype=data_type,
                    initializer=flow.constant_initializer(0),
                )
                x = x + values_blob
                y = flow.clip_by_value(x, min, max)
                flow.losses.add_loss(y)

            flow.watch_diff(x, grad_cb)
            return y

    else:

        def clip(values_blob):
            with flow.device_prior_placement(device_type, "0:0"):
                return flow.clip_by_value(values_blob, min, max, name="Clip")

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)
    if grad_cb is not None:
        func_config.train.primary_lr(1e-3)
        func_config.train.model_update_conf(dict(naive_conf={}))

    if dynamic:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.function(func_config)
        def clip_fn(values_def=flow.MirroredTensorDef(values.shape, dtype=data_type)):
            return clip(values_def)

        check_point = flow.train.CheckPoint()
        check_point.init()
        return clip_fn([values]).get().ndarray_list()[0]
        # return clip_fn([values]).get().ndarray()

    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.function(func_config)
        def clip_fn(values_def=flow.FixedTensorDef(values.shape, dtype=data_type)):
            return clip(values_def)

        check_point = flow.train.CheckPoint()
        check_point.init()
        return clip_fn(values).get().ndarray()


def _compare_with_tf(test_case, values, min, max, device_type, dynamic):
    x = tf.Variable(values)
    with tf.GradientTape() as t:
        y = tf.clip_by_value(x, min, max)
        dy = t.gradient(y, x)

    def compare_dy(dy_blob):
        test_case.assertTrue(
            np.array_equal(dy.numpy(), dy_blob.ndarray_list()[0] if dynamic else dy_blob.ndarray())
        )

    of_y = _of_clip_by_value(
        values=values,
        min=min,
        max=max,
        device_type=device_type,
        dynamic=dynamic,
        compare_fn=compare_dy,
    )
    test_case.assertTrue(np.array_equal(y.numpy(), of_y))


def test_clip_by_value(test_case):
    values = np.random.randint(low=-100, high=100, size=(8, 512, 4)).astype(np.float32)
    np_out = np.clip(values, -50, 50)

    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgList(arg_dict):
        of_out = _of_clip_by_value(values, -50, 50, *arg)
        test_case.assertTrue(np.array_equal(np_out, of_out))


def test_clip_by_value_grad(test_case):
    values = np.random.standard_normal(1024).astype(np.float32)
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["dynamic"] = [True, False]
    for arg in GenArgList(arg_dict):
        _compare_with_tf(test_case, values, 0, 0.5, *arg)
