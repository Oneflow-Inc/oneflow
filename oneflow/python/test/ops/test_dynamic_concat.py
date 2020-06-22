import numpy as np
import oneflow as flow

from oneflow.python.framework.local_blob import LocalFixedTensor, LocalMirroredTensor


def _of_dynamic_concat(
    inputs,
    input_static_shape,
    axis,
    device_type,
    watch_cb=None,
    make_watch_diff_cb=None,
):
    assert isinstance(inputs, (list, tuple))
    assert len(inputs) >= 2
    assert callable(make_watch_diff_cb)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.device_prior_placement(device_type, "0:0"))
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def dynamic_concat_job(
        input_0_def=flow.MirroredTensorDef(shape=input_static_shape, dtype=flow.float),
        input_1_def=flow.MirroredTensorDef(shape=input_static_shape, dtype=flow.float),
    ):
        var_0 = flow.get_variable(
            "Var0",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.constant_initializer(value=1, dtype=flow.float),
            trainable=True,
        )
        var_1 = flow.get_variable(
            "Var1",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.constant_initializer(value=1, dtype=flow.float),
            trainable=True,
        )
        if callable(watch_cb):
            flow.watch(var_0, watch_cb)
            flow.watch(var_1, watch_cb)
            flow.watch(flow.identity(input_0_def), watch_cb)
            flow.watch(flow.identity(input_1_def), watch_cb)

        var_0 = var_0 * input_0_def
        var_1 = var_1 * input_1_def
        if callable(watch_cb):
            flow.watch(var_0, watch_cb)
            flow.watch(var_1, watch_cb)

        result = flow.dynamic_concat([var_0, var_1], axis=axis)
        flow.losses.add_loss(result)
        flow.watch_diff(var_0, make_watch_diff_cb(0))
        flow.watch_diff(var_1, make_watch_diff_cb(1))
        return result

    check_point = flow.train.CheckPoint()
    check_point.init()
    ret = dynamic_concat_job([inputs[0]], [inputs[1]]).get()
    return ret.ndarray_list()[0]


def _test_dynamic_concat(test_case, output_shape, axis, device_type, verbose=False):
    assert output_shape[axis] > 2

    low = np.random.randint(low=1, high=output_shape[axis] - 1)
    high = np.random.randint(low=low, high=output_shape[axis])
    rand_output = np.random.rand(*output_shape).astype(np.single)
    slice_list_0 = []
    slice_list_1 = []
    for i in range(len(output_shape)):
        if i == axis:
            slice_list_0.append(slice(0, low))
            slice_list_1.append(slice(low, high))
        else:
            slice_list_0.append(slice(None))
            slice_list_1.append(slice(None))

    input_0 = rand_output[tuple(slice_list_0)]
    input_1 = rand_output[tuple(slice_list_1)]
    inputs = [input_0, input_1]

    def print_blob(blob):
        if isinstance(blob, LocalFixedTensor):
            print(blob.ndarray(), blob.ndarray().shape)
        elif isinstance(blob, LocalMirroredTensor):
            print(blob.ndarray_list()[0], blob.ndarray_list()[0].shape)
        else:
            raise NotImplementedError

    def make_watch_diff_cb(input_idx):
        def watch_diff_cb(blob):
            if isinstance(blob, LocalMirroredTensor):
                test_case.assertTrue(
                    np.array_equal(
                        blob.ndarray_list()[0],
                        np.ones(shape=inputs[input_idx].shape, dtype=np.single),
                    )
                )
            else:
                raise NotImplementedError

        return watch_diff_cb

    of_output = _of_dynamic_concat(
        inputs,
        tuple(output_shape),
        axis,
        device_type,
        print_blob if verbose else None,
        make_watch_diff_cb,
    )
    slice_list_all = []
    for i in range(len(output_shape)):
        if i == axis:
            slice_list_all.append(slice(0, high))
        else:
            slice_list_all.append(slice(None))
    exp_output = rand_output[tuple(slice_list_all)]
    if verbose:
        print("inputs shapes:", [input.shape for input in inputs])
        print("of_output shape:", of_output.shape)
        print("exp_output shape:", exp_output.shape)
        print("of_output:\n", of_output)
        print("exp_output:\n", exp_output)
    test_case.assertTrue(np.array_equal(of_output, exp_output))


def test_dynamic_concat_case_0(test_case):
    _test_dynamic_concat(test_case, (64, 4), 0, "gpu")
