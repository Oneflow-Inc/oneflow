import oneflow as flow
import numpy as np
import torch
import math


def test_gelu(test_case):
    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def GeluJob(x=flow.input_blob_def((10,))):
        return flow.keras.activations.gelu(x)

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_out = GeluJob(x).get()
        torch_out = gelu(torch.tensor(x)).numpy()
        test_case.assertTrue(np.allclose(of_out, torch_out, rtol=1e-3, atol=1e-4))
