from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _random_inputs(data_shape, segment_ids_shape, axis, num_segments):
    data = np.random.rand(*data_shape).astype(np.float32)
    segment_ids = np.random.randint(
        low=0, high=num_segments, size=segment_ids_shape, dtype=np.int32
    )
    return data, segment_ids


def _make_unsorted_segment_sum_fn(
    data, segment_ids, axis, num_segments, device_type, mirrored, compare_fn
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    if mirrored:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.train.primary_lr(1e-3)
    func_config.train.model_update_conf(dict(naive_conf={}))

    def do_unsorted_segment_sum(x_blob, i_blob):
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "data",
                shape=data.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            x = x + x_blob
            y = flow.math.unsorted_segment_sum(
                x, i_blob, axis=axis, num_segments=num_segments
            )
            flow.losses.add_loss(y)
        flow.watch_diff(x, compare_fn)
        return y

    if mirrored:

        @flow.global_function(func_config)
        def unsorted_segment_sum_fn(
            data_def=flow.MirroredTensorDef(data.shape, dtype=flow.float),
            segment_ids_def=flow.MirroredTensorDef(segment_ids.shape, dtype=flow.int32),
        ):
            return do_unsorted_segment_sum(data_def, segment_ids_def)

    else:

        @flow.global_function(func_config)
        def unsorted_segment_sum_fn(
            data_def=flow.FixedTensorDef(data.shape, dtype=flow.float),
            segment_ids_def=flow.FixedTensorDef(segment_ids.shape, dtype=flow.int32),
        ):
            return do_unsorted_segment_sum(data_def, segment_ids_def)

    return unsorted_segment_sum_fn


def _compare_unsorted_segment_sum_with_tf(
    test_case,
    device_type,
    data_shape,
    segment_ids_shape,
    axis,
    num_segments,
    mirrored=False,
):
    data, segment_ids = _random_inputs(
        data_shape, segment_ids_shape, axis, num_segments
    )
    i = tf.constant(segment_ids)
    with tf.GradientTape() as t:
        x = tf.Variable(data)
        y = tf.math.unsorted_segment_sum(x, i, num_segments=num_segments)

    dy = t.gradient(y, x)
    if isinstance(dy, tf.IndexedSlices):
        test_case.assertTrue(
            np.array_equal(segment_ids.ravel(), dy.segment_ids.numpy().ravel())
        )
        zero_data = tf.Variable(np.full(data.shape, 0.0, dtype=np.float32))
        dy = tf.math.tensor_scatter_nd_add(zero_data, i, dy.values)
    if mirrored:

        def compare_dy(data_grad):
            test_case.assertTrue(
                np.array_equal(dy.numpy(), data_grad.ndarray_list()[0])
            )

    else:

        def compare_dy(data_grad):
            test_case.assertTrue(np.array_equal(dy.numpy(), data_grad.ndarray()))

    unsorted_segment_sum_fn = _make_unsorted_segment_sum_fn(
        data, segment_ids, axis, num_segments, device_type, mirrored, compare_dy
    )

    check_point = flow.train.CheckPoint()
    check_point.init()

    if mirrored:
        of_y = unsorted_segment_sum_fn([data], [segment_ids]).get().ndarray_list()[0]
    else:
        of_y = unsorted_segment_sum_fn(data, segment_ids).get().ndarray()
    test_case.assertTrue(np.allclose(y.numpy(), of_y, rtol=1e-5, atol=1e-5))


def test_unsorted_segment_sum(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["data_shape"] = [(300, 1, 4)]
    arg_dict["segment_ids_shape"] = [(300)]
    arg_dict["axis"] = [0]
    arg_dict["num_segments"] = [2]

    for arg in GenArgList(arg_dict):
        _compare_unsorted_segment_sum_with_tf(test_case, *arg)


def test_unsorted_segment_sum_case_1(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["data_shape"] = [(200, 10, 20)]
    arg_dict["segment_ids_shape"] = [(200)]
    arg_dict["axis"] = [0]
    arg_dict["num_segments"] = [5]
    for arg in GenArgList(arg_dict):
        _compare_unsorted_segment_sum_with_tf(test_case, *arg)
