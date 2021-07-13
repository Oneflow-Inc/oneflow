import unittest
import oneflow.experimental as flow 
from automated_test_util import *

@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConv3DModule(flow.unittest.TestCase):
    def test_conv3d(test_case):
        for device in ["cpu", "cuda"]:
            test_module_against_pytorch(
                test_case, 
                "nn.Conv3d", 
                device=device, 
                n=2, 
                extra_generators={
                    "input": random_tensor(ndim=5, dim0=2, dim1=4, dim2=4, dim3=6, dim4=6),
                    "in_channels": constant(4),
                    "out_channels": constant(8),
                    "kernel_size": random(1, 3),
                    "stride": random(1, 2),
                    "padding": random(1, 2),
                    "dilation": constant(1),
                    "groups": constant(1),
                    "padding_mode": constant("zeros"), 
                    "bias": constant(True), 
                }
            )

    def test_conv3d_group(test_case):
        for device in ["cpu", "cuda"]:
            test_module_against_pytorch(
                test_case, 
                "nn.Conv3d", 
                device=device, 
                n=1, 
                extra_generators={
                    "input": random_tensor(ndim=5, dim0=2, dim1=4, dim2=4, dim3=6, dim4=6),
                    "in_channels": constant(4),
                    "out_channels": constant(8),
                    "kernel_size": random(1, 3),
                    "stride": random(1, 2),
                    "padding": random(1, 2),
                    "dilation": constant(1),
                    "groups": constant(2),
                    "padding_mode": constant("zeros"),
                },
            )
    
    def test_conv3d_depthwise(test_case):
        for device in ["cpu", "cuda"]:
            test_module_against_pytorch(
                test_case, 
                "nn.Conv3d", 
                n=1, 
                device=device, 
                extra_generators={
                    "input": random_tensor(ndim=5, dim0=2, dim1=4, dim2=4, dim3=6, dim4=6),
                    "in_channels": constant(4),
                    "out_channels": constant(8),
                    "kernel_size": random(1, 3),
                    "stride": random(1, 2),
                    "padding": random(1, 2),
                    "dilation": constant(1),
                    "groups": constant(4),
                    "padding_mode": constant("zeros"),
                },
            )

if __name__ == "__main__":
    unittest.main()