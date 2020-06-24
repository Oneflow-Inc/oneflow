import oneflow as flow
import numpy as np
import os


def test_lazy_input_output(test_case):
    flow.clear_default_session()
    flow.enable_eager_execution(False)

    @flow.global_function()
    def foo_job(input_def=flow.FixedTensorDef(shape=(2, 5))):
        var = flow.get_variable(
            name="var",
            shape=(2, 5),
            dtype=flow.float,
            initializer=flow.ones_initializer(),
        )
        output = var + input_def
        return output

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()
    input = np.arange(10).reshape(2, 5).astype(np.single)
    ret = foo_job(input).get()
    output = input + np.ones(shape=(2, 5), dtype=np.single)
    test_case.assertTrue(np.array_equal(output, ret.ndarray()))


def test_eager_output(test_case):
    if os.getenv("ENABLE_USER_OP") != "True":
        return

    flow.clear_default_session()
    flow.enable_eager_execution()

    @flow.global_function()
    def foo_job():
        x = flow.constant(1, shape=(2, 5), dtype=flow.float)
        # print(x.numpy_mirrored_list())
        return x

    # foo_job()
    ret = foo_job().get()
    test_case.assertTrue(
        np.array_equal(np.ones(shape=(2, 5), dtype=np.single), ret.ndarray_list()[0])
    )
