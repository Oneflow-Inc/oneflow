from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export

import collections

@oneflow_export("nn.conv2d_transpose")
def deconv2d(
    value=None,
    filter=None,
    strides=None,
    padding=None,
    data_format='NHWC',
    name=None,
    input=None,
    filters=None,
    dilations=1,
    output_padding=None,
):
    r"""2d transposed convolution
    Args:
    value: 4-d `Blob`
    filter: filter of transposed convolution, usually a variable
    output_shape: Not supported yet
    strides: `int` or `int list` Stride of the convolution
    padding: zero-padding will be added to both sides of each dimension in the input.
    data_format: `'NHWC'` or `'NCHW'`
    name: This operator's name
    input: Alias for value
    filters: Alias for filter
    dilations: Spacing between kernel element
    output_padding: Additional size added to one side of each dimension in the output shape.
    Returns:
    A `Blob` with the same type as `value`.
    Raises:
    ValueError: shapes of `filter` and `input` must match.
    """
    assert (value is not None) ^ (
        input is not None), "only one of `input` and `value` could be not None"
    assert (filter is not None) ^ (
        filters is not None), "only one of `filter` and `filters` could be not None"

    filters = filters or filter
    input = input or value

    assert len(input.static_shape) == 4
    assert len(filters.static_shape) == 4

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # data format
    if data_format.upper() == "NCHW":
        input_shape = input.static_shape[2:]
        kernel_size = filters.static_shape[2:4]
        channel_pos = "channels_first"
    elif data_format.upper() == "NHWC":
        input_shape = input.static_shape[1:3]
        kernel_size = filters.static_shape[-3:-1]
        channel_pos = "channels_last"
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    # dilations
    if dilations is None:
        dilations = [1, 1]
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 2, ValueError(
                "dilations length must be 2 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations, dilations]
        else:
            raise ValueError("dilations must be an int or a list.")

    # output_padding
    assert output_padding is not None
    if isinstance(output_padding, (list, tuple)):
        assert len(output_padding) == 2, ValueError(
            "output_padding length must be 2 when passed as a list."
        )
    elif isinstance(output_padding, int):
        output_padding = [output_padding, output_padding]
    else:
        raise ValueError("output_padding must be an int or a list.")

    assert (output_padding[0] >= 0) and (output_padding[0] < strides[0]) and \
            (output_padding[1] >= 0) and (output_padding[1] < strides[1]), \
            ValueError("output_padding value should be in range [0, stride]")

    if isinstance(padding, (list, tuple)):
        assert len(padding) == 2, ValueError(
            "padding length must be 2 when passed as a list."
        )
        padding_needed = [int(padding[0] * 2), int(padding[1] * 2)]
    elif isinstance(padding, int):
        padding_needed = 2 * padding
        padding_needed = [padding_needed, padding_needed]
    else:
        raise ValueError("padding must be an int or a list.")
    
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name",
            name if name is not None else id_util.UniqueStr("Deconv2d_"))
    op_conf.deconv_conf.x = input.logical_blob_name
    op_conf.deconv_conf.y = "out"
    op_conf.deconv_conf.weight = filters.logical_blob_name
    op_conf.deconv_conf.conv_conf.padding = "valid" # actually not uesd
    op_conf.deconv_conf.conv_conf.data_format = channel_pos
    if channel_pos == "channels_first":
        op_conf.deconv_conf.filters = filters.static_shape[1]
        op_conf.deconv_conf.conv_conf.kernel_size.extend(
            filters.static_shape[2:4])
    elif channel_pos == "channels_last":
        op_conf.deconv_conf.filters = filters.static_shape[3]
        op_conf.deconv_conf.conv_conf.kernel_size.extend(
            filters.static_shape[-3:-1])
    else:
        raise ValueError("invalid data_format")

    op_conf.deconv_conf.output_padding.extend(output_padding)
    op_conf.deconv_conf.conv_conf.strides.extend(strides)
    op_conf.deconv_conf.conv_conf.dilation_rate.extend(dilations)
    op_conf.deconv_conf.conv_conf.num_spatial_dims = 2
    op_conf.deconv_conf.conv_conf.torch_style_padding_conf.padding_needed.extend(padding_needed)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.conv2d_transpose_V2")
def deconv2d_tf(
    value=None,
    filter=None,
    output_shape=None,
    strides=None,
    padding=None,
    data_format='NCHW',
    name=None,
    input=None,
    filters=None,
    dilations=1,
):
    assert (value is not None) ^ (
        input is not None), "only one of `input` and `value` could be not None"
    assert (filter is not None) ^ (
        filters is not None), "only one of `filter` and `filters` could be not None"
    filters = filters or filter
    input = input or value
    assert len(input.static_shape) == 4
    assert len(filters.static_shape) == 4

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # data format
    if data_format.upper() == "NCHW":
        input_shape = input.static_shape[2:]
        kernel_size = filters.static_shape[2:4]
        assert output_shape is not None
        assert len(output_shape) == 4
        output_shape = output_shape[2:4]
    else:
        raise ValueError('data_format must be NCHW".')

    # dilations
    if dilations is None:
        dilations = [1, 1]
    else:
        if isinstance(dilations, (list, tuple)):
            assert len(dilations) == 2, ValueError(
                "dilations length must be 2 when passed as a list."
            )
        elif isinstance(dilations, int):
            dilations = [dilations, dilations]
        else:
            raise ValueError("dilations must be an int or a list.")
    # output_padding
    output_padding = [0, 0]
    if padding.upper() == "SAME":
        padding_needed = [0, 0]
        for i in range(2):
            assert (output_shape[i] + strides[i] - 1) // strides[i] == input_shape[i]
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            padding_needed[i] = max(0, (input_shape[i] - 1) * strides[i] \
                                + effective_filter_size - output_shape[i])
            tmp_output_size = (input_shape[i] - 1) * strides[i] + effective_filter_size - padding_needed[i]
            output_padding[i] = output_shape[i] - tmp_output_size
    elif padding.upper() == "VALID":
        padding_needed = [0, 0]
        for i in range(2):
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            assert (output_shape[i] + strides[i] - effective_filter_size) // strides[i] == input_shape[i]
            tmp_output_size = (input_shape[i] - 1) * strides[i] + effective_filter_size
            output_padding[i] = output_shape[i] - tmp_output_size
    else:
        raise ValueError('padding must be "SAME" or "VALID".')
    
    need_pad = False
    padding_one = [0, 0]
    for i in range(2):
        if padding_needed[i] % 2 != 0:
            padding_one[i] = 1
            padding_needed[i] = (padding_needed[i] - 1) / 2
            need_pad = True 
        else:
            padding_needed[i] = padding_needed[i] / 2  
    if need_pad:
        out = oneflow.nn.conv2d_transpose(input, filters, strides=strides, output_padding=output_padding, 
                                       dilations=dilations, padding=padding_needed, data_format="NCHW")
        out = oneflow.pad_grad(out, [(0,0),(0,0),(0,padding_one[0]),(0,padding_one[1])])
    else:
        out = oneflow.nn.conv2d_transpose(input, filters, strides=strides, output_padding=output_padding, 
                                       dilations=dilations, padding=padding_needed, data_format="NCHW")
    return out
