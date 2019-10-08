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


if __name__ == "__main__":
    test_case_1()
