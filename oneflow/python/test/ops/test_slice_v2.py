import numpy as np
import oneflow as flow


def _run_slice(input, index_args, dynamic=False, dtype=flow.float, input_shape=None):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(dtype)

    if input_shape is None:
        input_shape = input.shape

    def do_slice(x, indices):
        outputs = []
        for slice_tup_list in indices:
            output = flow.slice_v2(x, slice_tup_list)
            outputs.append(output)
        return outputs

    if dynamic is True:
        func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

        @flow.global_function(func_config)
        def slice(input_blob=flow.MirroredTensorDef(shape=input_shape, dtype=dtype)):
            return do_slice(input_blob, index_args)

        outputs = slice([input]).get()
        return map(lambda x: x.ndarray_list()[0], outputs)

    else:
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

        @flow.global_function(func_config)
        def slice(input_blob=flow.FixedTensorDef(shape=input_shape, dtype=dtype)):
            return do_slice(input_blob, index_args)

        outputs = slice(input).get()
        return map(lambda x: x.ndarray(), outputs)


def _check(test_case, ref, out):
    for _ref, _out in zip(ref, out):
        test_case.assertTrue(np.allclose(_ref, _out))


def test_slice_into_two_parts(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 0:2, :], input[:, 2:, :]]
    args = [
        [(None, None, None), (0, 2, None), (None, None, None)],
        [(None, None, None), (2, None, None), (None, None, None)],
    ]
    outputs = _run_slice(input, args)
    _check(test_case, results, outputs)


def test_slice_at_first_dim(test_case):
    input = np.random.rand(4, 5, 4).astype(np.float32)
    results = [input[2:None, :, :]]
    args = [[(2, None, None)]]
    outputs = _run_slice(input, args)
    _check(test_case, results, outputs)


def test_slice_at_two_dims(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 0:2, 2:]]
    args = [[(None, None, None), (0, 2, None), (2, None, None)]]
    outputs = _run_slice(input, args)
    _check(test_case, results, outputs)


def test_slice_with_collapse_dims(test_case):
    input = np.random.rand(2, 5, 4, 4, 3).astype(np.float32)
    results = [input[:, 0:2, :, :, 1:None]]
    args = [
        [
            (None, None, None),
            (0, 2, None),
            (None, None, None),
            (None, None, None),
            (1, None, None),
        ]
    ]
    outputs = _run_slice(input, args)
    _check(test_case, results, outputs)


def test_dynamic_slice(test_case):
    input = np.random.rand(2, 4, 4).astype(np.float32)
    results = [input[:, 1:, :]]
    args = [[(None, None, None), (1, None, None)]]
    outputs = _run_slice(input, args, dynamic=True, input_shape=(2, 5, 5))
    _check(test_case, results, outputs)


def test_dynamic_slice_case2(test_case):
    input = np.random.rand(2, 6, 3).astype(np.float32)
    results = [input[:, 1:, :]]
    args = [[(None, None, None), (1, None, None)]]
    outputs = _run_slice(input, args, dynamic=True, input_shape=(2, 5, 5))
    _check(test_case, results, outputs)


def test_dynamic_slice_at_two_dims(test_case):
    input = np.random.rand(2, 3, 2, 2).astype(np.float32)
    results = [input[:, 2:, :, 1:]]
    args = [[(None, None, None), (2, None, None), (None, None, None), (1, None, None)]]
    outputs = _run_slice(input, args, dynamic=True, input_shape=(2, 5, 3, 3))
    _check(test_case, results, outputs)


def test_dynamic_slice_at_first_dim_and_anthor_dim(test_case):
    input = np.random.rand(3, 6, 3, 3).astype(np.float32)
    results = [input[1:, :, :, 1:]]
    args = [[(1, None, None), (None, None, None), (None, None, None), (1, None, None)]]
    outputs = _run_slice(input, args, dynamic=True, input_shape=(4, 5, 5, 3))
    _check(test_case, results, outputs)


def test_slice_with_stride2(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 1::2, :]]
    args = [[(None, None, None), (1, None, 2)]]
    outputs = _run_slice(input, args, dynamic=True)
    _check(test_case, results, outputs)


def test_slice_at_two_dim_with_stride_more_than_one(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 1::3, ::2]]
    args = [[(None, None, None), (1, None, 3), (None, None, 2)]]
    outputs = _run_slice(input, args, dynamic=True)
    _check(test_case, results, outputs)


def test_slice_with_neg_index(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 2:-2, :]]
    args = [[(None, None, None), (2, -2, None)]]
    outputs = _run_slice(input, args, dynamic=True)
    _check(test_case, results, outputs)


def test_slice_with_neg_stride(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, 3:-4:-1, :]]
    args = [[(None, None, None), (3, -4, -1)]]
    outputs = _run_slice(input, args, dynamic=True)
    _check(test_case, results, outputs)


def test_slice_with_neg_stride2(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    results = [input[:, -1:1:-2, :]]
    args = [[(None, None, None), (-1, 1, -2)]]
    outputs = _run_slice(input, args, dynamic=True)
    _check(test_case, results, outputs)


def test_slice_grad(test_case):
    input = np.random.rand(2, 5, 4).astype(np.float32)
    ref = np.zeros(input.shape, dtype=np.float32)
    ref[:, 2:-2, :] = np.ones(input[:, 2:-2, :].shape, dtype=np.float32)

    def slice_grad_cb(dx_blob):
        dx = dx_blob.ndarray()
        test_case.assertTrue(np.allclose(ref, dx))

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.train.primary_lr(1e-3)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def slice(input_blob=flow.FixedTensorDef(shape=(2, 5, 4), dtype=flow.float)):
        x = flow.get_variable(
            shape=(2, 5, 4),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(2),
            name="variable",
        )
        x = flow.identity(x)
        flow.watch_diff(x, slice_grad_cb)
        y = flow.slice_v2(x, [(None, None, None), (2, -2, None)])
        flow.losses.add_loss(y)
        return y

    slice(input).get()
