import oneflow as flow
import numpy as np


def _of_broadcast_to_compatible_with(x, compatible_shape, x_shape=None):
    assert isinstance(compatible_shape, (list, tuple))
    if x_shape is None:
        x_shape = x.shape

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def broadcast_to_compatible_with_fn(
        x_def=flow.MirroredTensorDef(shape=x_shape, dtype=flow.float)
    ):
        compatible_var = [
            flow.get_variable(
                "compatible_var_{}".format(i),
                shape=cp_shape,
                dtype=flow.float,
                initializer=flow.random_normal_initializer(),
                trainable=False,
            )
            for i, cp_shape in enumerate(compatible_shape)
        ]
        return flow.broadcast_to_compatible_with(x_def, compatible_var)

    return broadcast_to_compatible_with_fn([x]).get().ndarray_list()[0]


def _of_broadcast_to_compatible_with_dynamic(x, a, b, x_shape=None, a_shape=None, b_shape=None):
    if x_shape is None:
        x_shape = x.shape

    if a_shape is None:
        a_shape = a.shape

    if b_shape is None:
        b_shape = b.shape

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def broadcast_to_compatible_with_fn(
        x_def=flow.MirroredTensorDef(x_shape, dtype=flow.float),
        a_def=flow.MirroredTensorDef(a_shape, dtype=flow.float),
        b_def=flow.MirroredTensorDef(b_shape, dtype=flow.float),
    ):
        return flow.broadcast_to_compatible_with(
            x_def, [flow.identity(a_def), flow.identity(b_def)]
        )

    return broadcast_to_compatible_with_fn([x], [a], [b]).get().ndarray_list()[0]


def _of_broadcast_to_compatible_with_grad(x, compatible_shape, dx_watcher):
    assert isinstance(compatible_shape, (list, tuple))
    assert callable(dx_watcher)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.train.primary_lr(1e-3)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def broadcast_to_compatible_with_fn(x_def=flow.FixedTensorDef(x.shape, dtype=flow.float)):
        x_var = flow.get_variable(
            "x_var",
            shape=x.shape,
            dtype=flow.float,
            initializer=flow.constant_initializer(0),
            trainable=True,
        )
        compatible_var = [
            flow.get_variable(
                "compatible_var_{}".format(i),
                shape=cp_shape,
                dtype=flow.float,
                initializer=flow.random_normal_initializer(),
                trainable=False,
            )
            for i, cp_shape in enumerate(compatible_shape)
        ]
        x_var = x_var + x_def
        y = flow.broadcast_to_compatible_with(x_var, compatible_var)
        flow.losses.add_loss(y)

        flow.watch_diff(x_var, dx_watcher)
        return y

    return broadcast_to_compatible_with_fn(x).get().ndarray()


def test_broadcast_to_compatible_with(test_case):
    x = np.random.standard_normal((5, 2)).astype(np.float32)
    compatible_shape = [[4, 5, 2], [4, 5, 1]]
    ret = _of_broadcast_to_compatible_with(x, compatible_shape)
    expected_ret = np.broadcast_to(x, [4, 5, 2])
    test_case.assertTrue(np.array_equal(expected_ret, ret))


def test_dynamic_broadcast_to_compatible_with(test_case):
    x = np.random.standard_normal((10, 6)).astype(np.float32)
    x_static_shape = (15, 6)
    a = np.random.standard_normal((3, 10, 6)).astype(np.float32)
    a_static_shape = (3, 15, 6)
    b = np.random.standard_normal((3, 10, 1)).astype(np.float32)
    b_static_shape = (3, 15, 1)
    ret = _of_broadcast_to_compatible_with_dynamic(
        x, a, b, x_static_shape, a_static_shape, b_static_shape
    )
    expected_ret = np.broadcast_to(x, [3, 10, 6])
    test_case.assertTrue(np.array_equal(expected_ret, ret))


def test_dynamic_broadcast_to_compatible_with_case_2(test_case):
    x = np.random.standard_normal((20, 1, 1)).astype(np.float32)
    x_static_shape = (23, 1, 1)
    a = np.random.standard_normal((11, 1)).astype(np.float32)
    a_static_shape = (15, 1)
    b = np.random.standard_normal((7,)).astype(np.float32)
    b_static_shape = (8,)
    ret = _of_broadcast_to_compatible_with_dynamic(
        x, a, b, x_static_shape, a_static_shape, b_static_shape
    )
    expected_ret = np.broadcast_to(x, [20, 11, 7])
    test_case.assertTrue(np.array_equal(expected_ret, ret))


def test_broadcast_to_compatible_with_grad(test_case):
    x = np.random.standard_normal((7, 1, 4)).astype(np.float32)
    compatible_shape = [[7, 1, 4], [5, 4]]

    def compare_dy(dx_blob):
        dx = np.ones([7, 5, 4], dtype=np.float32).sum(axis=1).reshape(x.shape)
        test_case.assertTrue(np.array_equal(dx, dx_blob.ndarray()))

    ret = _of_broadcast_to_compatible_with_grad(x, compatible_shape, compare_dy)
    exp_ret = np.broadcast_to(x, [7, 5, 4])
    test_case.assertTrue(np.array_equal(exp_ret, ret))


def test_broadcast_to_compatible_with_no_broadcast(test_case):
    x = np.random.standard_normal((9, 9, 6)).astype(np.float32)
    x_static_shape = (10, 9, 6)
    compatible_shape = [[6], [9, 1]]
    ret = _of_broadcast_to_compatible_with(x, compatible_shape, x_static_shape)
    test_case.assertTrue(np.array_equal(x, ret))
