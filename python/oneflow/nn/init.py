from oneflow.ops.initializer_util import CalcGain


def calculate_gain(nonlinearity, param=None):
    return CalcGain(nonlinearity, param)


def uniform_(tensor, a=0.0, b=1.0):
    tensor.uniform_(a, b)


def normal_(tensor, mean=0.0, std=1.0):
    tensor.normal_(mean, std)


def xavier_uniform_(tensor, gain=1.0, *, data_format="NCHW"):
    tensor.xavier_uniform_(gain, data_format=data_format)


def xavier_normal_(tensor, gain=1.0, *, data_format="NCHW"):
    tensor.xavier_normal_(gain, data_format=data_format)


def kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    tensor.kaiming_uniform_(a, mode, nonlinearity, data_format=data_format)


def kaiming_normal_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    tensor.kaiming_normal_(a, mode, nonlinearity, data_format=data_format)


def constant_(tensor, val):
    tensor.fill_(val)


def ones_(tensor):
    tensor.fill_(1)


def zeros_(tensor):
    tensor.fill_(0)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.ndimension() > 2:
        for s in tensor.size()[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return (fan_in, fan_out)


from oneflow.nn.init import calculate_gain
from oneflow.nn.init import uniform_
from oneflow.nn.init import normal_
from oneflow.nn.init import xavier_uniform_
from oneflow.nn.init import xavier_normal_
from oneflow.nn.init import kaiming_uniform_
from oneflow.nn.init import kaiming_normal_
from oneflow.nn.init import constant_
from oneflow.nn.init import ones_
from oneflow.nn.init import zeros_
