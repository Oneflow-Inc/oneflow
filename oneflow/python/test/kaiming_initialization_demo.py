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
    pytorch_var = None
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

    np.allclose(np.var(of_var), np.var(pytorch_var))
    np.allclose(np.mean(of_var), np.mean(pytorch_var))


if __name__ == "__main__":
    args = [
        [(5000, 6000), (50, 60, 70, 80)],
        ["random_normal", "random_uniform"],
        ["fan_in", "fan_out"],
        [None, "tanh", "sigmoid", "relu", "leaky_relu"],
        [1.0],
        [None, "channels_first"],
    ]
    arg_list = []
    for arg in itertools.product(*args):
        test_kaiming_initialization(*arg)
