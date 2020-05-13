from __future__ import absolute_import

import os
import random
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow
from oneflow.python.oneflow_export import oneflow_export

import collections
import os
import oneflow as flow

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

    if os.getenv("ENABLE_USER_OP") == 'True':
        if channel_pos == "channels_first":
            kernel_size_list = filters.static_shape[2:4]
        elif channel_pos == "channels_last":
            kernel_size_list = filters.static_shape[-3:-1]
        else:
            raise ValueError("invalid data_format")
        assert(isinstance(kernel_size_list, tuple))
        assert isinstance(groups, int)
        assert groups > 0
        if groups > 1:
            if data_format.upper() == "NCHW":
                assert groups <= filters.static_shape[0]
                assert filters.static_shape[0] % groups == 0
                assert groups <= input.static_shape[1]
                assert input.static_shape[1] % groups == 0
                assert filters.static_shape[1] == input.static_shape[1] // groups
            elif data_format.upper() == "NHWC":
                raise ValueError("data_format NHWC not support groups > 1")
            else:
                raise ValueError("invalid data_format")
        return (
                flow.user_op_builder(name if name is not None else id_util.UniqueStr("Conv2d_"))
                .Op("conv2d")
                .Input("in", [input])
                .Input("weight", [filters])
                .Output("out")
                .SetAttr("filters", filters.static_shape[0], "AttrTypeInt32")
                .SetAttr("padding", padding.lower(), "AttrTypeString")
                .SetAttr("data_format", channel_pos, "AttrTypeString")
                .SetAttr("kernel_size", kernel_size_list, "AttrTypeListInt32")
                .SetAttr("strides", strides, "AttrTypeListInt32")
                .SetAttr("dilation_rate", dilations, "AttrTypeListInt32")
                .SetAttr("groups", groups, "AttrTypeInt32")
                .Build()
                .InferAndTryRun()
                .RemoteBlobList()[0]
        )
    else:
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

        assert isinstance(groups, int)
        assert groups > 0
        if groups > 1:
            if data_format.upper() == "NCHW":
                assert groups <= filters.static_shape[0]
                assert filters.static_shape[0] % groups == 0
                assert groups <= input.static_shape[1]
                assert input.static_shape[1] % groups == 0
                assert filters.static_shape[1] == input.static_shape[1] // groups
            elif data_format.upper() == "NHWC":
                raise ValueError("data_format NHWC not support groups > 1")
            else:
                raise ValueError("invalid data_format")
        op_conf.conv_2d_conf.groups = groups

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
    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("MaxPool2D_"))
            .Op("max_pool_2d")
            .Input("x", [input])
            .Output("y")
        )
        assert padding in ["VALID", "SAME"]
        op.SetAttr("padding", padding.lower(), "AttrTypeString")
        assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
        data_format = "channels_last" if data_format == "NHWC" else "channels_first"
        op.SetAttr("data_format", data_format, "AttrTypeString")
        pool_size = _GetSequence(ksize, 2, "ksize")
        op.SetAttr("pool_size", pool_size, "AttrTypeListInt32")
        strides = _GetSequence(strides, 2, "strides")
        op.SetAttr("strides", strides, "AttrTypeListInt32")
        return (
            op
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
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
    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("AvgPool2D_"))
            .Op("avg_pool_2d")
            .Input("x", [input])
            .Output("y")
        )
        assert padding in ["VALID", "SAME"]
        op.SetAttr("padding", padding.lower(), "AttrTypeString")
        assert data_format in ["NHWC", "NCHW", "NCHW_VECT_C"]
        data_format = "channels_last" if data_format == "NHWC" else "channels_first"
        op.SetAttr("data_format", data_format, "AttrTypeString")
        pool_size = _GetSequence(ksize, 2, "ksize")
        op.SetAttr("pool_size", pool_size, "AttrTypeListInt32")
        strides = _GetSequence(strides, 2, "strides")
        op.SetAttr("strides", strides, "AttrTypeListInt32")
        return (
            op
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
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
    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("MaxPool3D_"))
            .Op("max_pool_3d")
            .Input("x", [input])
            .Output("y")
        )
        assert padding in ["VALID", "SAME"]
        op.SetAttr("padding", padding.lower(), "AttrTypeString")
        assert data_format in ["NDHWC", "NCDHW"]
        data_format = "channels_last" if data_format == "NHWC" else "channels_first"
        op.SetAttr("data_format", data_format, "AttrTypeString")
        pool_size = _GetSequence(ksize, 3, "ksize")
        op.SetAttr("pool_size", pool_size, "AttrTypeListInt32")
        strides = _GetSequence(strides, 3, "strides")
        op.SetAttr("strides", strides, "AttrTypeListInt32")
        return (
            op
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
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
    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("AvgPool3D_"))
            .Op("avg_pool_3d")
            .Input("x", [input])
            .Output("y")
        )
        assert padding in ["VALID", "SAME"]
        op.SetAttr("padding", padding.lower(), "AttrTypeString")
        assert data_format in ["NDHWC", "NCDHW"]
        data_format = "channels_last" if data_format == "NHWC" else "channels_first"
        op.SetAttr("data_format", data_format, "AttrTypeString")
        pool_size = _GetSequence(ksize, 3, "ksize")
        op.SetAttr("pool_size", pool_size, "AttrTypeListInt32")
        strides = _GetSequence(strides, 3, "strides")
        op.SetAttr("strides", strides, "AttrTypeListInt32")
        return (
            op
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()[0]
        )
    else:
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


def _softmax_need_transpose(x, axis):
    assert type(axis) is int
    dim_num = len(x.shape)
    assert dim_num >= 2
    if axis < 0: axis += dim_num
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
    if axis is None:
        axis = -1

    if os.getenv("ENABLE_USER_OP") != 'True':
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

    need_transpose, permute = _softmax_need_transpose(logits, axis)
    if need_transpose:
        logits = oneflow.transpose(logits, perm=permute)

    out = (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Softmax_"))
        .Op("softmax")
        .Input("in", [logits])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    if need_transpose:
        out = oneflow.transpose(out, perm=permute)
    return out


@oneflow_export("nn.softmax_grad")
def softmax_grad(y, dy, axis=None, name=None):
    if axis is None:
        axis = -1

    if os.getenv("ENABLE_USER_OP") != 'True':
        assert type(axis) is int
        op_conf = op_conf_util.OperatorConf()

        name_prefix = name if name is not None else id_util.UniqueStr("SoftmaxGrad_")
        setattr(op_conf, "name", name_prefix)

        need_transpose, permute = _softmax_need_transpose(y, axis)

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

    need_transpose, permute = _softmax_need_transpose(logits, axis)
    if need_transpose:
        y = oneflow.transpose(y, perm=permute)
        dy = oneflow.transpose(dy, perm=permute)

    dx = (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Softmax_"))
        .Op("softmax_grad")
        .Input("y", [y])
        .Input("dy", [dy])
        .Output("dx")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    if need_transpose:
        dx = oneflow.transpose(dx, perm=permute)
    return dx

@oneflow_export("nn.sparse_cross_entropy")
def sparse_cross_entropy(
    labels=None, prediction=None, name=None
):
    assert labels is not None
    assert prediction is not None

    if os.getenv("ENABLE_USER_OP") == 'True':
        if len(labels.shape) == len(prediction.shape):
            assert labels.shape[-1] == 1
            labels = flow.squeeze(labels, axis = [-1])
        else:
            assert len(labels.shape) == len(prediction.shape) - 1

        return (
        flow.user_op_builder(name if name is not None else id_util.UniqueStr("SparseCrossEntropy_"))
        .Op("sparse_cross_entropy")
        .Input("prediction", [prediction])
        .Input("label", [labels])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
        )
    else:
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

    if os.getenv("ENABLE_USER_OP") == 'True':
        if len(labels.shape) == len(logits.shape):
            assert labels.shape[-1] == 1
            labels = flow.squeeze(labels, axis = [-1])
        else:
            assert len(labels.shape) == len(logits.shape) - 1

        prob, out = (
            flow.user_op_builder(name if name is not None else id_util.UniqueStr("SparseSoftmaxCrossEntropy_"))
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

def random_mask_like(like, rate, seed=None, noise_shape=None, name=None):
    assert rate is not None and rate >= 0.0 and rate < 1.0
    mask_op = (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("RandomMaskLike_"))
        .Op("random_mask_like")
        .Input("like", [like])
        .Output("out")
        .SetAttr("rate", float(rate), "AttrTypeFloat")
    )
    if seed is not None:
        mask_op.SetAttr("seed", seed, "AttrTypeInt64")
    else:
        mask_op.SetAttr("seed", random.randint(-2**63 + 1, 2**63 - 1), "AttrTypeInt64")
        
    if noise_shape is not None:
        assert 0, "noise_shape will be supported later."
        assert isinstance(noise_shape, (list, tuple))
    return (
        mask_op
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

@oneflow_export("nn.dropout")
def dropout(x, noise_shape=None, seed=None, name=None, rate=None):
    if os.getenv("ENABLE_USER_OP") != 'True':
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
    
    assert rate is not None and rate >= 0.0 and rate < 1.0
    mask = random_mask_like(x, rate, seed, noise_shape) 
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Dropout_"))
        .Op("dropout")
        .Input("in", [x])
        .Input("mask", [mask])
        .Output("out")
        .SetAttr("scale", float(1.0 / (1.0 - rate)), "AttrTypeFloat")
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
    padding='SAME',
    data_format='NHWC',
    name=None,
    input=None,
    filters=None,
    dilations=None
):
    r"""2d transposed convolution
    Args:
    value: 4-d `Blob`
    filter: filter of transposed convolution, usually a variable
    output_shape: Not supported yet
    strides: `int` or `int list`
    padding: `'VALID'` or `'SAME'`
    data_format: `'NHWC'` or `'NCHW'`
    name: This operator's name
    input: Alias for value
    filters: Alias for filter
    dilations: Not supported yet
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
    assert output_shape is None, "output_shape not supported yet"
    assert dilations is None, "dilations not supported yet"
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

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name",
            name if name is not None else id_util.UniqueStr("Deconv2d_"))
    op_conf.deconv_conf.x = input.logical_blob_name
    op_conf.deconv_conf.y = "out"
    op_conf.deconv_conf.filter = filters.logical_blob_name
    op_conf.deconv_conf.conv_conf.padding = padding.lower()
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

    op_conf.deconv_conf.conv_conf.strides.extend(strides)
    op_conf.deconv_conf.conv_conf.dilation_rate.extend(dilations)
    op_conf.deconv_conf.use_bias = False
    op_conf.deconv_conf.conv_conf.num_spatial_dims = 2
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("nn.leaky_relu")
def leaky_relu(x, alpha=0.2, name=None):
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("LeakyRelu_"))
        .Op("leaky_relu")
        .Input("x", [x])
        .Output("y")
        .SetAttr("alpha", float(alpha), "AttrTypeFloat")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
