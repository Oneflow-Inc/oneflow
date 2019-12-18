import oneflow as of
import numpy as np


def test_case_1():
    r"""dim0 slice"""
    func_config = of.FunctionConfig()
    func_config.default_data_type(of.float)

    @of.function(func_config)
    def slice_dim0_job(input_blob=of.FixedTensorDef(shape=(2, 5, 4), dtype=of.float32)):
        part1 = of.slice(input_blob, (None, 0, 0), (None, 2, None))
        part2 = of.slice(input_blob, (None, 2, 0), (None, 3, None))
        return part1, part2

    input = np.random.rand(2, 5, 4).astype(np.float32)
    p1, p2 = slice_dim0_job(input).get()

    ref1 = input[:, :2, :]
    ref2 = input[:, 2:5, :]
    print("input: \n", input)
    print("part1: \n", ref1)
    print("part2: \n", ref2)
    assert np.allclose(p1.ndarray(), ref1)
    assert np.allclose(p2.ndarray(), ref2)


if __name__ == "__main__":
    test_case_1()
