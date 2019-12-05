import oneflow as flow
import numpy as np
import torch
import math


def _test_relu(device_type):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def ReluJob(x=flow.input_blob_def((10,))):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.keras.activations.relu(x)

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_ret = ReluJob(x).get()
        expected = x * (r > 0)
        assert np.allclose(expected, of_ret)


def test_relu(test_case):
    _test_relu("gpu")
    _test_relu("cpu")


def _test_gelu(device_type):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def GeluJob(x=flow.input_blob_def((10,))):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.keras.activations.gelu(x)

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_out = GeluJob(x).get()
        torch_out = gelu(torch.tensor(x)).numpy()
        assert np.allclose(of_out, torch_out, rtol=1e-3, atol=1e-4)


def test_gelu(test_case):
    _test_gelu("gpu")
    _test_gelu("cpu")


def _test_sigmoid(device_type):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def SigmoidJob(x=flow.input_blob_def((10,))):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.keras.activations.sigmoid(x)

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_out = SigmoidJob(x).get()
        torch_out = torch.sigmoid(torch.Tensor(x)).numpy()
        assert np.allclose(of_out, torch_out)


def test_sigmoid(test_case):
    _test_sigmoid("gpu")
    _test_sigmoid("cpu")


def _test_tanh(device_type):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def TanHJob(x=flow.input_blob_def((10,))):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.keras.activations.tanh(x)

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_out = TanHJob(x).get()
        torch_out = torch.tanh(torch.Tensor(x)).numpy()
        assert np.allclose(of_out, torch_out)


def test_tanh(test_case):
    _test_tanh("gpu")
    _test_tanh("cpu")
