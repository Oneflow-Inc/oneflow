from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.ops.user_op_builder as user_op_builder
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("data.OFRecordRawDecoder", "data.ofrecord_raw_decoder")
def OFRecordRawDecoder(
        input_blob,
        blob_name,
        shape,
        dtype,
        dim1_varying_length=False,
        auto_zero_padding=False,
        name=None):
    if name is None:
        name = id_util.UniqueStr("OFRecordRawDecoder_")
    return flow.user_op_builder(name)\
            .Op("ofrecord_raw_decoder")\
            .Input("in",[input_blob])\
            .Output("out")\
            .SetAttr("name", blob_name, "AttrTypeString")\
            .SetAttr("shape", shape, "AttrTypeShape")\
            .SetAttr("data_type", dtype, "AttrTypeInt64")\
            .SetAttr("dim1_varying_length", dim1_varying_length, "AttrTypeBool")\
            .SetAttr("auto_zero_padding", auto_zero_padding, "AttrTypeBool")\
            .Build().InferAndTryRun().RemoteBlobList()[0]

@oneflow_export("data.OFRecordImageDecoderRandomCrop", "data.ofrecord_image_decoder_random_crop")
def OFRecordImageDecoderRandomCrop(
        input_blob,
        blob_name,
        color_space="BGR",
        num_attempts=10,
        seed=None,
        random_area=[0.08, 1.0],
        random_aspect_ratio=[0.75, 1.333333],
        name=None):
    if name is None:
        name = id_util.UniqueStr("OFRecordImageDecoderRandomCrop_")
    return flow.user_op_builder(name)\
            .Op("ofrecord_image_decoder_random_crop")\
            .Input("in",[input_blob])\
            .Output("out")\
            .SetAttr("name", blob_name, "AttrTypeString")\
            .SetAttr("color_space", color_space, "AttrTypeString")\
            .SetAttr("num_attempts", num_attempts, "AttrTypeInt32")\
            .SetRandomSeed(seed)\
            .SetAttr("random_area", random_area, "AttrTypeListFloat")\
            .SetAttr("random_aspect_ratio", random_aspect_ratio, "AttrTypeListFloat")\
            .Build().InferAndTryRun().RemoteBlobList()[0]

@oneflow_export("image.Resize", "image.resize")
def Resize(
        input_blob,
        color_space="BGR",
        interp_type="Linear",
        resize_x=0,
        resize_y=0,
        name=None):
    if name is None:
        name = id_util.UniqueStr("ImageResize_")
    return flow.user_op_builder(name)\
            .Op("image_resize")\
            .Input("in",[input_blob])\
            .Output("out")\
            .SetAttr("color_space", color_space, "AttrTypeString")\
            .SetAttr("interp_type", interp_type, "AttrTypeString")\
            .SetAttr("resize_x", resize_x, "AttrTypeInt64")\
            .SetAttr("resize_y", resize_y, "AttrTypeInt64")\
            .Build().InferAndTryRun().RemoteBlobList()[0]

@oneflow_export("image.CropMirrorNormalize", "image.crop_mirror_normalize")
def CropMirrorNormalize(
        input_blob,
        mirror_blob=None,
        color_space="BGR",
        output_layout="NCHW",
        mean=[0.0],
        std=[1.0],
        output_dtype=flow.float,
        name=None):
    if name is None:
        name = id_util.UniqueStr("CropMirrorNormalize_")
    op = flow.user_op_builder(name)\
            .Op("crop_mirror_normalize")\
            .Input("in",[input_blob])
    if mirror_blob is not None:
        op = op.Input("mirror", [mirror_blob])
    return op.Output("out")\
            .SetAttr("color_space", color_space, "AttrTypeString")\
            .SetAttr("output_layout", output_layout, "AttrTypeString")\
            .SetAttr("mean", mean, "AttrTypeListFloat")\
            .SetAttr("std", std, "AttrTypeListFloat")\
            .SetAttr("output_dtype", output_dtype, "AttrTypeInt32")\
            .Build().InferAndTryRun().RemoteBlobList()[0]

@oneflow_export("random.CoinFlip", "random.coin_flip")
def CoinFlip(
        batch_size=1,
        seed=None,
        probability=0.5,
        name=None):
    if name is None:
        name = id_util.UniqueStr("CoinFlip_")
    return flow.user_op_builder(name)\
            .Op("coin_flip")\
            .Output("out")\
            .SetAttr("batch_size", batch_size, "AttrTypeInt64")\
            .SetAttr("probability", probability, "AttrTypeFloat")\
            .SetRandomSeed(seed)\
            .Build().InferAndTryRun().RemoteBlobList()[0]


