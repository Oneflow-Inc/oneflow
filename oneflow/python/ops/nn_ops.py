from __future__ import absolute_import

import collections
import os
import random

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("nn.conv2d")
def conv2d(
    input,
    filters,
    strides,
    padding,
    data_format="NHWC",
    dilations=None,
    groups=1,
    name=None,
):
    r"""2d convolution 

    Analogous to `tf.nn.conv2d <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d>`_

    """
    name = name if name is not None else id_util.UniqueStr("Conv2d_")
    assert len(input.shape) == 4
    assert len(filters.shape) == 4

    NDims = 2
    if isinstance(strides, (list, tuple)):
        assert len(strides) == 2, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    if data_format.upper() != "NCHW" and data_format.upper() != "NHWC":
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"

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

    if channel_pos == "channels_first":
        input_size = input.shape[2:4]
        kernel_size_list = filters.shape[2:4]
    elif channel_pos == "channels_last":
        input_size = input.shape[-3:-1]
        kernel_size_list = filters.shape[-3:-1]
    else:
        raise ValueError("invalid data_format")

    if padding.upper() == "SAME":
        padding_left = [0] * NDims
        padding_right = [0] * NDims
        for i in range(NDims):
            effective_filter_size = (kernel_size_list[i] - 1) * dilations[i] + 1
            tmp_output_size = (input_size[i] + strides[i] - 1) // strides[i]
            padding_needed = max(
                0,
                (tmp_output_size - 1) * strides[i]
                + effective_filter_size
                - input_size[i],
            )
            padding_left[i] = padding_needed // 2
            padding_right[i] = padding_needed - padding_needed // 2
        if padding_left != padding_right:
            assert data_format.upper() == "NCHW"
            input = flow.pad(
                input,
                [
                    (0, 0),
                    (0, 0),
                    (padding_left[0], padding_right[0]),
                    (padding_left[1], padding_right[1]),
                ],
                name=name + "_pad" if name is not None else None,
            )
            padding_needed = [0] * NDims
        else:
            padding_needed = [p * 2 for p in padding_left]
    elif padding.upper() == "VALID":
        padding_needed = [0] * NDims
    else:
        raise ValueError('padding must be "SAME" or "VALID".')

    assert isinstance(kernel_size_list, tuple)
    assert isinstance(groups, int)
    assert groups > 0
    if groups > 1:
        if data_format.upper() == "NCHW":
            assert groups <= filters.shape[0]
            assert filters.shape[0] % groups == 0
            assert groups <= input.shape[1]
            assert input.shape[1] % groups == 0
            assert filters.shape[1] == input.shape[1] // groups
        elif data_format.upper() == "NHWC":
            raise ValueError("data_format NHWC not support groups > 1")
        else:
            raise ValueError("invalid data_format")
    return (
        flow.user_op_builder(name)
        .Op("conv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", filters.shape[0], "AttrTypeInt32")
        .Attr("padding_needed", padding_needed, "AttrTypeListInt32")
        .Attr("data_format", channel_pos, "AttrTypeString")
        .Attr("kernel_size", kernel_size_list, "AttrTypeListInt32")
        .Attr("strides", strides, "AttrTypeListInt32")
        .Attr("dilation_rate", dilations, "AttrTypeListInt32")
        .Attr("groups", groups, "AttrTypeInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.batch_normalization")
def batch_normalization(
    x, mean, variance, offset, scale, variance_epsilon, axis=-1, name=None
):
    r"""
    This op does not fully align with tf.nn.batch_normalization. mean, variable, offset and scale
    are always 1D. Users need to specify "axis" to 1 for NCHW data format.

    """

    assert axis >= -len(x.shape) and axis < len(x.shape)
    if axis < 0:
        axis += len(x.shape)

    if name is None:
        name = id_util.UniqueStr("BatchNorm_")

    builder = (
        flow.user_op_builder(name)
        .Op("normalization")
        .Input("x", [x])
        .Input("moving_mean", [mean])
        .Input("moving_variance", [variance])
        .Input("gamma", [scale])
        .Input("beta", [offset])
        .Output("y")
        .Attr("axis", axis, "AttrTypeInt32")
        .Attr("epsilon", variance_epsilon, "AttrTypeFloat")
        .Attr("training", False, "AttrTypeBool")
        # momentum is not used
        .Attr("momentum", 0.0, "AttrTypeFloat")
    )
    return builder.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.bias_add")
def bias_add(value, bias, data_format=None, name=None):
    r"""
    Analogous to `tf.nn.bias_add <https://www.tensorflow.org/api_docs/python/tf/nn/bias_add>`_

    """
    # TODO: name unused, fix it
    if name is None:
        name = id_util.UniqueStr("BiasAdd_")

    if data_format is None:
        bias_add_axis = 1
    else:
        if data_format.startswith("NC"):
            bias_add_axis = 1
        elif data_format.startswith("N") and data_format.endswith("C"):
            bias_add_axis = len(value.shape) - 1
        else:
            raise ValueError("data_format must be of the form `N...C` or `NC...`")

    return (
        flow.user_op_builder(name)
        .Op("bias_add")
        .Input("a", [value])
        .Input("b", [bias])
        .Output("out")
        .Attr("axis", bias_add_axis, "AttrTypeInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


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
    r"""
    Analogous to `tf.nn.max_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool2d>`_

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool2D_")
        )
        .Op("max_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower(), "AttrTypeString")
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format, "AttrTypeString")
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size, "AttrTypeListInt32")
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides, "AttrTypeListInt32")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool2d")
def avg_pool2d(input, ksize, strides, padding, data_format="NHWC", name=None):
    r"""
    Analogous to `tf.nn.avg_pool2d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool2d>`_

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool2D_")
        )
        .Op("avg_pool_2d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower(), "AttrTypeString")
    assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format, "AttrTypeString")
    pool_size = _GetSequence(ksize, 2, "ksize")
    op.Attr("pool_size", pool_size, "AttrTypeListInt32")
    strides = _GetSequence(strides, 2, "strides")
    op.Attr("strides", strides, "AttrTypeListInt32")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.max_pool3d")
def max_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    r"""
    Analogous to `tf.nn.max_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/max_pool3d>`_

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("MaxPool3D_")
        )
        .Op("max_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower(), "AttrTypeString")
    assert data_format in ["NDHWC", "NCDHW"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format, "AttrTypeString")
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size, "AttrTypeListInt32")
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides, "AttrTypeListInt32")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.avg_pool3d")
def avg_pool3d(input, ksize, strides, padding, data_format="NDHWC", name=None):
    r"""
    Analogous to `tf.nn.avg_pool3d <https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool3d>`_

    """
    op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("AvgPool3D_")
        )
        .Op("avg_pool_3d")
        .Input("x", [input])
        .Output("y")
    )
    assert padding in ["VALID", "SAME"]
    op.Attr("padding", padding.lower(), "AttrTypeString")
    assert data_format in ["NDHWC", "NCDHW"]
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    op.Attr("data_format", data_format, "AttrTypeString")
    pool_size = _GetSequence(ksize, 3, "ksize")
    op.Attr("pool_size", pool_size, "AttrTypeListInt32")
    strides = _GetSequence(strides, 3, "strides")
    op.Attr("strides", strides, "AttrTypeListInt32")
    return op.Build().InferAndTryRun().RemoteBlobList()[0]


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0:
        axis += dim_num
    assert axis >= 1
    assert axis < dim_num

    need_transpose = False
    permute = [i for i in range(dim_num)]
    if axis > 0 and axis != dim_num - 1:
        need_transpose = True
        permute[axis] = permute[-1]
        permute[-1] = axis
    return need_transpose, permute


@oneflow_export("nn.softmax")
def softmax(logits, axis=None, name=None):
    r"""
    Analogous to `tf.nn.softmax <https://www.tensorflow.org/api_docs/python/tf/nn/softmax>`_

    """
    if axis is None:
        axis = -1

    need_transpose, permute = _softmax_need_transpose(logits, axis)
    if need_transpose:
        logits = flow.transpose(logits, perm=permute)

    out = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Softmax_")
        )
        .Op("softmax")
        .Input("in", [logits])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    if need_transpose:
        out = flow.transpose(out, perm=permute)
    return out


@oneflow_export("nn.softmax_grad")
def softmax_grad(y, dy, axis=None, name=None):
    if axis is None:
        axis = -1

    need_transpose, permute = _softmax_need_transpose(y, axis)
    if need_transpose:
        y = flow.transpose(y, perm=permute)
        dy = flow.transpose(dy, perm=permute)

    dx = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Softmax_")
        )
        .Op("softmax_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    if need_transpose:
        dx = flow.transpose(dx, perm=permute)
    return dx


@oneflow_export("nn.sparse_cross_entropy")
def sparse_cross_entropy(labels=None, prediction=None, name=None):
    assert labels is not None
    assert prediction is not None

    if len(labels.shape) == len(prediction.shape):
        assert labels.shape[-1] == 1
        labels = flow.squeeze(labels, axis=[-1])
    else:
        assert len(labels.shape) == len(prediction.shape) - 1

    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SparseCrossEntropy_")
        )
        .Op("sparse_cross_entropy")
        .Input("prediction", [prediction])
        .Input("label", [labels])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.softmax_cross_entropy_with_logits")
def softmax_cross_entropy_with_logits(labels=None, logits=None, name=None):
    r"""
    Analogous to `tf.nn.softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits>`_

    """

    assert labels is not None
    assert logits is not None

    assert labels.shape == logits.shape
    assert labels.dtype == logits.dtype

    prob, out = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("SoftmaxCrossEntropy_")
        )
        .Op("softmax_cross_entropy")
        .Input("prediction", [logits])
        .Input("label", [labels])
        .Output("prob")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return out


@oneflow_export("nn.sparse_softmax_cross_entropy_with_logits")
def sparse_softmax_cross_entropy_with_logits(labels=None, logits=None, name=None):
    r"""
    Analogous to `tf.nn.sparse_softmax_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits>`_

    """
    assert labels is not None
    assert logits is not None

    if os.getenv("ENABLE_USER_OP") != "False":
        if len(labels.shape) == len(logits.shape):
            assert labels.shape[-1] == 1
            labels = flow.squeeze(labels, axis=[-1])
        else:
            assert len(labels.shape) == len(logits.shape) - 1

        prob, out = (
            flow.user_op_builder(
                name
                if name is not None
                else id_util.UniqueStr("SparseSoftmaxCrossEntropy_")
            )
            .Op("sparse_softmax_cross_entropy")
            .Input("prediction", [logits])
            .Input("label", [labels])
            .Output("prob")
            .Output("out")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()
        )
        return out
    else:
        return sparse_cross_entropy(labels=labels, prediction=softmax(logits))


@oneflow_export("nn.sigmoid_cross_entropy_with_logits")
def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None):
    r"""
    Analogous to `tf.nn.sigmoid_cross_entropy_with_logits <https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits>`_

    """
    assert labels is not None
    assert logits is not None
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("SigmoidCrossEntropy_"),
    )
    op_conf.sigmoid_cross_entropy_conf.prediction = logits.unique_name
    op_conf.sigmoid_cross_entropy_conf.label = labels.unique_name
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
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


@oneflow_export("nn.random_mask_like")
def random_mask_like(like, rate, seed=None, noise_shape=None, name=None):
    assert rate is not None and rate >= 0.0 and rate < 1.0
    mask_op = (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("RandomMaskLike_")
        )
        .Op("random_mask_like")
        .Input("like", [like])
        .Output("out")
        .Attr("rate", float(rate), "AttrTypeFloat")
    )
    if seed is not None:
        mask_op.Attr("seed", seed, "AttrTypeInt64")
    else:
        mask_op.Attr(
            "seed", random.randint(-(2 ** 63) + 1, 2 ** 63 - 1), "AttrTypeInt64"
        )

    if noise_shape is not None:
        assert 0, "noise_shape will be supported later."
        assert isinstance(noise_shape, (list, tuple))
    return mask_op.Build().InferAndTryRun().RemoteBlobList()[0]


@oneflow_export("nn.dropout")
def dropout(x, noise_shape=None, seed=None, name=None, rate=None):
    r"""
    Analogous to `tf.nn.dropout <https://www.tensorflow.org/api_docs/python/tf/nn/dropout>`_

    """
    assert rate is not None and rate >= 0.0 and rate < 1.0
    if not flow.current_global_function_desc().IsTrainable() or rate == 0.0:
        return x
    mask = random_mask_like(x, rate, seed, noise_shape)
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Dropout_")
        )
        .Op("dropout")
        .Input("in", [x])
        .Input("mask", [mask])
        .Output("out")
        .Attr("scale", float(1.0 / (1.0 - rate)), "AttrTypeFloat")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.conv2d_transpose")
def deconv2d(
    value=None,
    filter=None,
    output_shape=None,
    strides=None,
    padding="VALID",
    data_format="NHWC",
    name=None,
    input=None,
    filters=None,
    dilations=None,
):
    r"""2d transposed convolution

    Args:
    value: 4-d `Blob`
    filter: filter of transposed convolution, usually a variable
    output_shape: A 1-D Tensor representing the output shape of the deconvolution op
    strides: `int` or `int list`
    padding: `'VALID'` or `'SAME'`
    data_format: `'NHWC'` or `'NCHW'`
    name: This operator's name
    input: Alias for value
    filters: Alias for filter
    dilations: The dilation factor for each dimension of input.
    Returns:
        A `Blob` with the same type as `value`.

    Raises:
        ValueError: shapes of `filter` and `input` must match.
    """
    assert (value is not None) ^ (
        input is not None
    ), "only one of `input` and `value` could be not None"
    assert (filter is not None) ^ (
        filters is not None
    ), "only one of `filter` and `filters` could be not None"
    filters = filters or filter
    input = input or value

    NDims = 2
    assert len(input.shape) == 2 + NDims
    assert len(filters.shape) == 2 + NDims
    assert len(output_shape) == 2 + NDims
    assert output_shape[0] == input.shape[0]

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

    # data format
    if data_format.upper() == "NCHW":
        input_shape = input.shape[2:]
        kernel_size = filters.shape[2:4]
        output_shape = output_shape[2:4]
        channels = filters.shape[1]
    elif data_format.upper() == "NHWC":
        input_shape = input.shape[1:3]
        kernel_size = filters.shape[-3:-1]
        output_shape = output_shape[1:3]
        channels = filters.shape[3]
        assert dilations == [1, 1], ValueError(
            "dialtions must be 1 when data format is NHWC "
        )
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # output_padding and padding_needed
    output_padding = [0] * NDims
    padding_needed = [0] * NDims
    if padding.upper() == "VALID":
        for i in range(NDims):
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            assert (output_shape[i] + strides[i] - effective_filter_size) // strides[
                i
            ] == input_shape[i]
            tmp_output_shape = (input_shape[i] - 1) * strides[i] + effective_filter_size
            output_padding[i] = output_shape[i] - tmp_output_shape
    elif padding.upper() == "SAME":
        padding_left = [0] * NDims
        padding_right = [0] * NDims
        for i in range(NDims):
            assert (output_shape[i] + strides[i] - 1) // strides[i] == input_shape[i]
            effective_filter_size = (kernel_size[i] - 1) * dilations[i] + 1
            padding_needed[i] = max(
                0,
                (input_shape[i] - 1) * strides[i]
                + effective_filter_size
                - output_shape[i],
            )
            tmp_output_shape = (
                (input_shape[i] - 1) * strides[i]
                + effective_filter_size
                - padding_needed[i]
            )
            output_padding[i] = output_shape[i] - tmp_output_shape
            padding_left[i] = padding_needed[i] // 2
            padding_right[i] = padding_needed[i] - padding_needed[i] // 2
    else:
        raise ValueError('padding must be "SAME" or "VALID".')
    # add pad op if needs odd padding
    if padding.upper() == "SAME" and padding_left != padding_right:
        assert data_format.upper() == "NCHW"
        padding_needed = [0] * NDims
        input = (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Conv2d_")
            )
            .Op("deconv2d")
            .Input("in", [input])
            .Input("weight", [filters])
            .Output("out")
            .Attr("filters", channels, "AttrTypeInt32")
            .Attr("padding_needed", padding_needed, "AttrTypeListInt32")
            .Attr("data_format", channel_pos, "AttrTypeString")
            .Attr("kernel_size", kernel_size, "AttrTypeListInt32")
            .Attr("strides", strides, "AttrTypeListInt32")
            .Attr("dilation_rate", dilations, "AttrTypeListInt32")
            .Attr("output_padding", output_padding, "AttrTypeListInt32")
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
        return flow.pad_grad(
            input,
            [
                (0, 0),
                (0, 0),
                (padding_left[0], padding_right[0]),
                (padding_left[1], padding_right[1]),
            ],
            name=name + "_pad_grad" if name is not None else None,
        )

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
        .Op("deconv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", channels, "AttrTypeInt32")
        .Attr("padding_needed", padding_needed, "AttrTypeListInt32")
        .Attr("data_format", channel_pos, "AttrTypeString")
        .Attr("kernel_size", kernel_size, "AttrTypeListInt32")
        .Attr("strides", strides, "AttrTypeListInt32")
        .Attr("dilation_rate", dilations, "AttrTypeListInt32")
        .Attr("output_padding", output_padding, "AttrTypeListInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.torch_conv2d_transpose")
def deconv2d_torch(
    value=None,
    filter=None,
    output_padding=None,
    strides=None,
    padding_needed=None,
    data_format="NHWC",
    name=None,
    input=None,
    filters=None,
    dilations=None,
):

    assert (value is not None) ^ (
        input is not None
    ), "only one of `input` and `value` could be not None"

    assert (filter is not None) ^ (
        filters is not None
    ), "only one of `filter` and `filters` could be not None"
    filters = filters or filter
    input = input or value

    NDims = 2
    assert len(input.shape) == 2 + NDims
    assert len(filters.shape) == 2 + NDims

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

    # data format
    if data_format.upper() == "NCHW":
        input_shape = input.shape[2:]
        kernel_size = filters.shape[2:4]
        channels = filters.shape[1]
    elif data_format.upper() == "NHWC":
        input_shape = input.shape[1:3]
        kernel_size = filters.shape[-3:-1]
        channels = filters.shape[3]
        assert dilations == [1, 1], ValueError(
            "dialtions must be 1 when data format is NHWC "
        )
    else:
        raise ValueError('data_format must be "NHWC" or "NCHW".')

    channel_pos = "channels_first" if data_format.startswith("NC") else "channels_last"

    # strides
    if isinstance(strides, (list, tuple)):
        assert len(strides) == NDims, ValueError(
            "strides length must be 2 when passed as a list."
        )
    elif isinstance(strides, int):
        strides = [strides, strides]
    else:
        raise ValueError("strides must be an int or a list.")

    # output_padding and padding_needed

    return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
        .Op("deconv2d")
        .Input("in", [input])
        .Input("weight", [filters])
        .Output("out")
        .Attr("filters", channels, "AttrTypeInt32")
        .Attr("padding_needed", padding_needed, "AttrTypeListInt32")
        .Attr("data_format", channel_pos, "AttrTypeString")
        .Attr("kernel_size", kernel_size, "AttrTypeListInt32")
        .Attr("strides", strides, "AttrTypeListInt32")
        .Attr("dilation_rate", dilations, "AttrTypeListInt32")
        .Attr("output_padding", output_padding, "AttrTypeListInt32")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("nn.leaky_relu")
def leaky_relu(x, alpha=0.2, name=None):
    return (
        flow.user_op_builder(
            name if name is not None else id_util.UniqueStr("LeakyRelu_")
        )
        .Op("leaky_relu")
        .Input("x", [x])
        .Output("y")
        .Attr("alpha", float(alpha), "AttrTypeFloat")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
