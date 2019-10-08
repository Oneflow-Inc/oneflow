import oneflow as of
import numpy as np


def test_case_1():
    r"""dim0 stack"""

    @of.function
    def stack_func(
        a_blob=of.input_blob_def(
            shape=(512, 4), dtype=of.float32, is_dynamic=True
        ),
        b_blob=of.input_blob_def(
            shape=(128, 4), dtype=of.float32, is_dynamic=True
        ),
    ):
        return of.stack([a_blob, b_blob], axis=0)

    a = np.random.rand(412, 4).astype(np.float32)
    b = np.random.rand(100, 4).astype(np.float32)
    c = np.concatenate((a, b), axis=0)
    of_c = stack_func(a, b).get().ndarray()
    print("stack result: \n", of_c)
    assert np.allclose(c, of_c)
    print("test case 1 Done!")


def test_case_2():
    r"""stack grad"""

    a = np.random.rand(412, 4).astype(np.float32)
    b = np.random.rand(100, 4).astype(np.float32)

    def a_diff_func(blob):
        global a_diff
        a_diff = blob.ndarray()
        print("a_diff shape: \n", a_diff.shape)
        assert np.allclose(
            a_diff.astype(np.float32), np.ones((412, 10), dtype=np.float32)
        )
        print("a_diff compare Done!")

    def b_diff_func(blob):
        global b_diff
        b_diff = blob.ndarray()
        print("b_diff shape: \n", b_diff.shape)
        assert np.allclose(
            b_diff.astype(np.float32), np.ones((100, 10), dtype=np.float32)
        )
        print("b_diff compare Done!")

    @of.function
    def stack_var_func(
        a_blob=of.input_blob_def(
            shape=(512, 4), dtype=of.float32, is_dynamic=True
        ),
        b_blob=of.input_blob_def(
            shape=(128, 4), dtype=of.float32, is_dynamic=True
        ),
    ):
        of.config.train.model_update_conf(dict(naive_conf={}))
        of.config.train.primary_lr(0.1)
        var_a = of.layers.dense(inputs=a_blob, units=10, use_bias=False)
        var_b = of.layers.dense(inputs=b_blob, units=10, use_bias=False)
        var_c = of.stack([var_a, var_b], axis=0)
        of.losses.add_loss(var_c)

        of.watch_diff(var_a, a_diff_func)
        of.watch_diff(var_b, b_diff_func)
        return var_c

    stack_var_func(a, b).get().ndarray()


if __name__ == "__main__":
    # test_case_1()
    test_case_2()
