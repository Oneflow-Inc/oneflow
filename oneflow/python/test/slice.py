import oneflow as of
import numpy as np


def test_case_1():
    r"""dim0 slice"""

    @of.function
    def slice_dim0_job(
        input_blob=of.input_blob_def(shape=(2, 10, 4), dtype=of.float32)
    ):
        part1 = of.slice(input_blob, (0, 0, 0), (1, -1, -1))
        part2 = of.slice(input_blob, (1, 0, 0), (1, -1, -1))
        return part1, part2

    input = np.random.rand(2, 10, 4).astype(np.float32)
    p1, p2 = slice_dim0_job(input).get()

    print("input: \n", input)
    print("part1: \n", input[0:1, :, :])
    print("part2: \n", input[1:, :, :])
    assert np.allclose(p1.ndarray(), input[0:1, :, :])
    assert np.allclose(p2.ndarray(), input[1:, :, :])


if __name__ == "__main__":
    test_case_1()
