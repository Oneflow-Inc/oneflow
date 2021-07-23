import unittest
import numpy as np
import oneflow as flow
from automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestMaskedFill(flow.unittest.TestCase):
    @unittest.skip("has bug now, need rewrite")
    def test_masked_fill_aginst_pytorch(test_case):
        import numpy as np
        import torch

        def mask_tensor(shape):
            def generator(_):
                rng = np.random.default_rng()
                np_arr = rng.integers(low=0, high=2, size=shape)
                return (
                    flow.Tensor(np_arr, dtype=flow.int8),
                    torch.tensor(np_arr, dtype=torch.bool),
                )

            return generator

        for device in ["cpu", "cuda"]:
            test_flow_against_pytorch(
                test_case,
                "masked_fill",
                extra_annotations={"mask": flow.Tensor, "value": float},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=4, dim1=5),
                    "mask": mask_tensor((4, 5)),
                    "value": constant(3.14),
                },
                device=device,
            )
            test_tensor_against_pytorch(
                test_case,
                "masked_fill",
                extra_annotations={"mask": flow.Tensor, "value": float},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=4, dim1=5),
                    "mask": mask_tensor((4, 5)),
                    "value": constant(3.14),
                },
                device=device,
            )


if __name__ == "__main__":
    unittest.main()
