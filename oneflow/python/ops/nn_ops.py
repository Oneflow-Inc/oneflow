from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export

import collections


@oneflow_export("nn.conv2d")
def conv2d(
    input,
    filters,
    strides,
    padding,
    data_format="NHWC",
    dilations=None,
    name=None,
):
    assert len(input.static_shape) == 4
    assert len(filters.static_shape) == 4

    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    if padding.upper() != "SAME" and padding.upper() != "VALID":
        raise ValueError('padding must be "SAME" or "VALID".')

    if data_format.upper() != "NCHW" and data_format.upper() != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    channel_pos = (
        "channels_first" if data_format.startswith("NC") else "channels_last"
    )

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

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name if name is not None else id_util.UniqueStr("Conv2d_"))
    setattr(op_conf.conv_2d_conf, "in", input.logical_blob_name)
    op_conf.conv_2d_conf.out = "out"
    op_conf.conv_2d_conf.weight = filters.logical_blob_name
    op_conf.conv_2d_conf.filters = filters.static_shape[0]
    op_conf.conv_2d_conf.padding = padding.lower()
    op_conf.conv_2d_conf.data_format = channel_pos
    if channel_pos == "channels_first":
        op_conf.conv_2d_conf.kernel_size.extend(filters.static_shape[2:4])
    elif channel_pos == "channels_last":
        op_conf.conv_2d_conf.kernel_size.extend(filters.static_shape[-3:-1])
    else:
        raise ValueError("invalid data_format")
    op_conf.conv_2d_conf.strides.extend(strides)
    op_conf.conv_2d_conf.dilation_rate.extend(dilations)
    op_conf.conv_2d_conf.use_bias = False

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("nn.bias_add")
def bias_add(value, bias, data_format=None, name=None):
    # TODO: name unused, fix it
    if name is None:
        name = id_util.UniqueStr("BiasAdd_")

    if data_format is None:
        bias_add_axis = 1
    else:
        if data_format.startswith("NC"):
            bias_add_axis = 1
        elif data_format.startswith("N") and data_format.endswith("C"):
            bias_add_axis = len(value.static_shape) - 1
        else:
            raise ValueError(
                "data_format must be of the form `N...C` or `NC...`"
            )

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.bias_add_conf, "a", value.logical_blob_name)
    setattr(op_conf.bias_add_conf, "b", bias.logical_blob_name)
    setattr(op_conf.bias_add_conf, "out", "out")
    setattr(op_conf.bias_add_conf, "axis", bias_add_axis)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.max_pool1d")
def max_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.avg_pool1d")
def avg_pool1d(input, ksize, strides, padding, data_format="NWC", name=None):
    # TODO: fix cuDNN bugs in pooling_1d
    raise NotImplementedError


@oneflow_export("nn.max_pool2d")
def max_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("MaxPool2D_"),
    )
    setattr(op_conf.max_pooling_2d_conf, "in", input.logical_blob_name)
    setattr(op_conf.max_pooling_2d_conf, "out", "out")
    op_conf.max_pooling_2d_conf.pool_size[:] = _GetSequence(ksize, 2, "ksize")
    op_conf.max_pooling_2d_conf.strides[:] = _GetSequence(strides, 2, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.max_pooling_2d_conf, "padding", padding)
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    setattr(
        op_conf.max_pooling_2d_conf,
        "data_format",
        "channels_last" if data_format == "NHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.avg_pool2d")
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AveragePool2D_"),
    )
    setattr(op_conf.average_pooling_2d_conf, "in", input.logical_blob_name)
    setattr(op_conf.average_pooling_2d_conf, "out", "out")
    op_conf.average_pooling_2d_conf.pool_size[:] = _GetSequence(
        ksize, 2, "ksize"
    )
    op_conf.average_pooling_2d_conf.strides[:] = _GetSequence(
        strides, 2, "strides"
    )
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.average_pooling_2d_conf, "padding", padding)
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    setattr(
        op_conf.average_pooling_2d_conf,
        "data_format",
        "channels_last" if data_format == "NHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.max_pool3d")
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("MaxPool3D_"),
    )
    setattr(op_conf.max_pooling_3d_conf, "in", input.logical_blob_name)
    setattr(op_conf.max_pooling_3d_conf, "out", "out")
    op_conf.max_pooling_3d_conf.pool_size[:] = _GetSequence(ksize, 3, "ksize")
    op_conf.max_pooling_3d_conf.strides[:] = _GetSequence(strides, 3, "strides")
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.max_pooling_3d_conf, "padding", padding)
    assert data_format in ["NDHWC", "NCDHW"]
    setattr(
        op_conf.max_pooling_3d_conf,
        "data_format",
        "channels_last" if data_format == "NDHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.avg_pool3d")
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("AveragePool3D_"),
    )
    setattr(op_conf.average_pooling_3d_conf, "in", input.logical_blob_name)
    setattr(op_conf.average_pooling_3d_conf, "out", "out")
    op_conf.average_pooling_3d_conf.pool_size[:] = _GetSequence(
        ksize, 3, "ksize"
    )
    op_conf.average_pooling_3d_conf.strides[:] = _GetSequence(
        strides, 3, "strides"
    )
    assert padding in ["VALID", "SAME"]
    setattr(op_conf.average_pooling_3d_conf, "padding", padding)
    assert data_format in ["NDHWC", "NCDHW"]
    setattr(
        op_conf.average_pooling_3d_conf,
        "data_format",
        "channels_last" if data_format == "NDHWC" else "channels_first",
    )
    compile_context.CurJobAddOp(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("nn.softmax")
def softmax(logits, axis=None, name=None):
    if axis is None:
        axis = -1
    assert type(axis) is int
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("Softmax_"),
    )
    setattr(op_conf.softmax_conf, "in", logits.logical_blob_name)
    op_conf.softmax_conf.axis = axis
    op_conf.softmax_conf.out = "out"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.softmax_grad")
def softmax_grad(y, dy, axis=None, name=None):
    if axis is None:
        axis = -1
    assert type(axis) is int
    op_conf = op_conf_util.OperatorConf()

    name_prefix = name if name is not None else id_util.UniqueStr("SoftmaxGrad_")
    setattr(op_conf, "name", name_prefix)

    need_transpose = False
    permute = [i for i in range(len(y.shape))]
    if axis > 0 and axis != len(y.shape) - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis

    if need_transpose:
        y = oneflow.transpose(y, perm=permute)
        dy = oneflow.transpose(dy, perm=permute)
    setattr(op_conf.softmax_grad_conf, "y", y.logical_blob_name)
    setattr(op_conf.softmax_grad_conf, "dy", dy.logical_blob_name)

    op_conf.softmax_grad_conf.axis = -1
    op_conf.softmax_grad_conf.dx = "dx"
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "dx"
    dx = remote_blob_util.RemoteBlob(lbi)

    if need_transpose:
        dx = oneflow.transpose(dx, perm=permute)
    return dx

@oneflow_export("nn.sparse_cross_entropy")
def sparse_cross_entropy(
    labels=None, prediction=None, name=None
):
    assert labels is not None
    assert prediction is not None
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SparseCrossEntropy_"),
    )
    setattr(
        op_conf.sparse_cross_entropy_conf,
        "prediction",
        prediction.logical_blob_name,
    )
    setattr(
        op_conf.sparse_cross_entropy_conf, "label", labels.logical_blob_name
    )
    setattr(op_conf.sparse_cross_entropy_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.sparse_softmax_cross_entropy_with_logits")
def sparse_softmax_cross_entropy_with_logits(
    labels=None, logits=None, name=None
):
    assert labels is not None
    assert logits is not None
    return sparse_cross_entropy(labels=labels, prediction=softmax(logits))

@oneflow_export("nn.sigmoid_cross_entropy_with_logits")
def sigmoid_cross_entropy_with_logits(
    labels=None, logits=None, name=None
):
    assert labels is not None
    assert logits is not None
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SigmoidCrossEntropy_"),
    )
    op_conf.sigmoid_cross_entropy_conf.prediction = logits.logical_blob_name
    op_conf.sigmoid_cross_entropy_conf.label = labels.logical_blob_name
    op_conf.sigmoid_cross_entropy_conf.loss = "loss"
    op_conf.sigmoid_cross_entropy_conf.label_type = labels.dtype
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "loss"
    return remote_blob_util.RemoteBlob(lbi)

def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]

    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(
                name, n, current_n
            )
        )


@oneflow_export("nn.dropout")
def dropout(x, noise_shape=None, seed=None, name=None, rate=None):
    # dropout op
    op_conf = op_conf_util.OperatorConf()
    if name is None:
        op_conf.name = id_util.UniqueStr("Dropout_")
    else:
        op_conf.name = name
    setattr(op_conf.dropout_conf, "in", x.logical_blob_name)
    setattr(op_conf.dropout_conf, "out", "out")
    # random mask like op
    mask_op_conf = op_conf_util.OperatorConf()
    mask_op_conf.name = "RandomMask4" + op_conf.name;
    setattr(mask_op_conf.random_mask_like_conf, "like", x.logical_blob_name)
    setattr(mask_op_conf.random_mask_like_conf, "out", "out")
    if noise_shape is not None:
        assert isinstance(noise_shape, (list, tuple))
        mask_op_conf.random_mask_like_conf.noise_shape.dim.extend(list(noise_shape))
    if seed is not None:
        setattr(mask_op_conf.random_mask_like_conf, "seed", seed)
    assert rate is not None and rate >= 0.0 and rate < 1.0
    setattr(mask_op_conf.random_mask_like_conf, "rate", rate)
    compile_context.CurJobAddOp(mask_op_conf)
    mask_lbi = logical_blob_id_util.LogicalBlobId()
    mask_lbi.op_name = mask_op_conf.name
    mask_lbi.blob_name = "out"
    mask_blob = remote_blob_util.RemoteBlob(mask_lbi)

    setattr(op_conf.dropout_conf, "mask", mask_blob.logical_blob_name)
    setattr(op_conf.dropout_conf, "scale", 1.0 / (1.0 - rate))

    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("nn.conv2d_transpose")
def deconv2d(
    value=None,
    filter=None,
    output_shape=None,
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

    # if output_shape is not None:
    #     assert len(output_shape) == 2
    #     output_padding = [0, 0]
    #     if padding.upper() == "SAME":
    #         # raise ValueError('SAME not supported')
    #         padding_needed = [0, 0]
    #         for i in range(2):
    #             assert (output_shape[i] + strides[i] - 1) // strides[i] == input_shape[i]
    #             effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
    #             padding_needed[i] = max(0, (input_shape[i] - 1) * strides[i] \
    #                                 + effective_filter_size - output_shape[i])
    #             tmp_output_size = (input_shape[i] - 1) * strides[i] + effective_filter_size - padding_needed[i]
    #             output_padding[i] = output_shape[i] - tmp_output_size

    #     elif padding.upper() == "VALID":
    #         padding_needed = [0, 0]
    #         for i in range(2):
    #             effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
    #             assert (output_shape[i] + strides[i] - effective_filter_size) // strides[i] == input_shape[i]
    #             tmp_output_size = (input_shape[i] - 1) * strides[i] + effective_filter_size
    #             output_padding[i] = output_shape[i] - tmp_output_size
    #     else:
    #         raise ValueError('padding must be "SAME" or "VALID".')
        
    
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name",
            name if name is not None else id_util.UniqueStr("Deconv2d_"))
    op_conf.deconv_conf.x = input.logical_blob_name
    op_conf.deconv_conf.y = "out"
    op_conf.deconv_conf.weight = filters.logical_blob_name
    op_conf.deconv_conf.conv_conf.padding = "torch"
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
    op_conf.deconv_conf.conv_conf.padding_needed.extend(padding_needed)
    # op_conf.deconv_conf.padding_needed.extend(padding_needed)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.conv2d_transpose2")
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
