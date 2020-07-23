from __future__ import absolute_import

import oneflow
import oneflow.python.framework.id_util as id_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("pad")
def pad(x, paddings, constant_value=0, name=None):
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("pad")
        .Input("x", [x])
        .Output("y")
        .Attr("padding_before", padding_before)
        .Attr("padding_after", padding_after)
        .Attr("floating_constant_value", float(constant_value))
        .Attr("integral_constant_value", int(constant_value))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("pad_grad")
def pad_grad(x, paddings, constant_value=0, name=None):
    padding_before = []
    padding_after = []
    if isinstance(paddings, (list, tuple)):
        assert len(paddings) == len(x.shape), ValueError(
            "paddings must be the same size of input dims"
        )
        for p in paddings:
            assert isinstance(p, (list, tuple)) and len(p) == 2, ValueError(
                "the elem of paddings must be a tuple or a list with length of 2"
            )
            padding_before.append(p[0])
            padding_after.append(p[1])
    else:
        raise ValueError("paddings must be a tuple or a list.")
    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("PadGrad_")
        )
        .Op("pad_grad")
        .Input("dy", [x])
        .Output("dx")
        .Attr("padding_before", padding_before)
        .Attr("padding_after", padding_after)
        .Attr("floating_constant_value", float(constant_value))
        .Attr("integral_constant_value", int(constant_value))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("same_padding")
def same_padding(
    x,
    padding,
    num_spatial_dims,
    data_format,
    kernel_size,
    strides,
    dilation_rate,
    constant_value=0,
    name=None,
):
    assert isinstance(padding, str) and (
        padding.upper() == "SAME_LOWER" or padding.upper() == "SAME_UPPER"
    ), 'padding must be "SAME_LOWER" or "SAME_UPPER".'
    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"

    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Pad_"))
        .Op("same_padding")
        .Input("x", [x])
        .Output("y")
        .Attr("padding", padding.upper())
        .Attr("num_spatial_dims", num_spatial_dims)
        .Attr("data_format", channel_pos)
        .Attr("kernel_size", kernel_size)
        .Attr("strides", strides)
        .Attr("dilation_rate", dilation_rate)
        .Attr("floating_constant_value", float(constant_value))
        .Attr("integral_constant_value", int(constant_value))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
