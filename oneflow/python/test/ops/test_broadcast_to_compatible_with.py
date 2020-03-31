import oneflow as flow
import numpy as np


def _of_broadcast_to_compatible_with(x, compatible_shape, x_static_shape=None):
    assert isinstance(compatible_shape, (list, tuple))

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def broadcast_to_compatible_with_fn(
        x_def=flow.MirroredTensorDef(x_static_shape or x.shape, dtype=flow.float)
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
