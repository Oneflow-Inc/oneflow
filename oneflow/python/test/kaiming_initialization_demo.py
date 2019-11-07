import oneflow as flow
import torch
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_conf_util
import itertools


flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)


def test_kaiming_initialization(
    shape, distribution, mode, nonlinearity, negative_slope, data_format
):
    @flow.function
    def GetVariableJob():
        var = flow.get_variable(
            name="var",
            shape=shape,
            dtype=data_type_conf_util.kFloat,
            initializer=flow.kaiming_initializer(
                shape=shape,
                distribution=distribution,
                mode=mode,
                nonlinearity=nonlinearity,
                negative_slope=negative_slope,
                data_format=data_format,
            ),
        )
        return var

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_var = GetVariableJob().get()

    # PyTorch
    if distribution == "random_normal":
        pytorch_var = torch.nn.init.kaiming_normal_(
            torch.empty(shape),
            a=negative_slope,
            mode=mode,
            nonlinearity="linear" if nonlinearity is None else nonlinearity,
        )
    elif distribution == "random_uniform":
        pytorch_var = torch.nn.init.kaiming_uniform_(
            torch.empty(shape),
            a=negative_slope,
            mode=mode,
            nonlinearity="linear" if nonlinearity is None else nonlinearity,
        )
    else:
        raise NotImplementedError(
            "Only support normal and uniform distribution"
        )
    pytorch_var = pytorch_var.detach().cpu().numpy()

    assert np.allclose(np.var(of_var), np.var(pytorch_var), rtol=1e-3, atol=1e-5)
    assert np.allclose(np.mean(of_var), np.mean(pytorch_var), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    # Only test cases in MaskRCNN, Cartesian product of
    # mode = ["fan_in", "fan_out"]
    # shape = [(5000, 6000), (50, 60, 70, 80)]
    # nonlinearity = ["leaky_relu", "relu"]
    # distribution = ["random_normal", "random_uniform"]
    test_kaiming_initialization((5000, 6000), "random_normal", "fan_in", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_uniform", "fan_in", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_normal", "fan_in", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_uniform", "fan_in", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_normal", "fan_in", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_uniform", "fan_in", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_normal", "fan_in", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_uniform", "fan_in", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_normal", "fan_out", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_uniform", "fan_out", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_normal", "fan_out", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((5000, 6000), "random_uniform", "fan_out", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_normal", "fan_out", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_uniform", "fan_out", "leaky_relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_normal", "fan_out", "relu", 1.0, "channels_first")
    # test_kaiming_initialization((50, 60, 70, 80), "random_uniform", "fan_out", "relu", 1.0, "channels_first")
