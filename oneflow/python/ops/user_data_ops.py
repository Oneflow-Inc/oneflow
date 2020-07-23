"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.common.data_type_pb2 as data_type_util

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.remote_blob import BlobDef
from typing import Sequence, Optional, Union


@oneflow_export("data.OFRecordRawDecoder", "data.ofrecord_raw_decoder")
def OFRecordRawDecoder(
    input_blob,
    blob_name,
    shape,
    dtype,
    dim1_varying_length=False,
    auto_zero_padding=False,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("OFRecordRawDecoder_")
    return (
        flow.user_op_builder(name)
        .Op("ofrecord_raw_decoder")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("name", blob_name)
        .Attr("shape", shape)
        .Attr("data_type", dtype)
        .Attr("dim1_varying_length", dim1_varying_length)
        .Attr("auto_zero_padding", auto_zero_padding)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export(
    "data.OFRecordImageDecoderRandomCrop", "data.ofrecord_image_decoder_random_crop"
)
def OFRecordImageDecoderRandomCrop(
    input_blob,
    blob_name,
    color_space="BGR",
    num_attempts=10,
    seed=None,
    random_area=[0.08, 1.0],
    random_aspect_ratio=[0.75, 1.333333],
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("OFRecordImageDecoderRandomCrop_")
    return (
        flow.user_op_builder(name)
        .Op("ofrecord_image_decoder_random_crop")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("name", blob_name)
        .Attr("color_space", color_space)
        .Attr("num_attempts", num_attempts)
        .SetRandomSeed(seed)
        .Attr("random_area", random_area)
        .Attr("random_aspect_ratio", random_aspect_ratio)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("data.OFRecordImageDecoder", "data.ofrecord_image_decoder")
def OFRecordImageDecoder(input_blob, blob_name, color_space="BGR", name=None):
    if name is None:
        name = id_util.UniqueStr("OFRecordImageDecoder_")
    return (
        flow.user_op_builder(name)
        .Op("ofrecord_image_decoder")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("name", blob_name)
        .Attr("color_space", color_space)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("image.Resize", "image.resize")
def image_resize(
    image: BlobDef,
    target_size: Union[int, Sequence[int]],
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    keep_aspect_ratio: bool = False,
    resize_side: str = "shorter",
    channels: int = 3,
    dtype: int = data_type_util.kUInt8,
    interpolation_type: str = "auto",
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("ImageResize_")

    if keep_aspect_ratio:
        assert isinstance(target_size, int)
        assert resize_side in (
            "shorter",
            "longer",
        ), 'resize_side must be "shorter" or "longer"'

        if min_size is None:
            min_size = 0

        if max_size is None:
            max_size = 0

        if resize_side == "shorter":
            resize_longer = False
        elif resize_side == "longer":
            resize_longer = True
        else:
            raise ValueError

        op = (
            flow.user_op_builder(name)
            .Op("image_resize_keep_aspect_ratio")
            .Input("in", [image])
            .Output("out")
            .Output("size")
            .Output("scale")
            .Attr("target_size", target_size)
            .Attr("min_size", min_size)
            .Attr("max_size", max_size)
            .Attr("resize_longer", resize_longer)
            .Attr("interpolation_type", interpolation_type)
            .Build()
        )
        res_image, new_size, scale = op.InferAndTryRun().RemoteBlobList()

    else:
        assert isinstance(target_size, (list, tuple))
        assert len(target_size) == 2
        target_w, target_h = target_size

        op = (
            flow.user_op_builder(name)
            .Op("image_resize_to_fixed")
            .Input("in", [image])
            .Output("out")
            .Output("scale")
            .Attr("target_width", target_w)
            .Attr("target_height", target_h)
            .Attr("channels", channels)
            .Attr("data_type", dtype)
            .Attr("interpolation_type", interpolation_type)
            .Build()
        )
        res_image, scale = op.InferAndTryRun().RemoteBlobList()
        new_size = None

    return res_image, scale, new_size


@oneflow_export("image.target_resize", "image_target_resize")
def image_target_resize(
    images: BlobDef,
    target_size: int,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    resize_side: str = "shorter",
    interpolation_type: str = "auto",
    name: Optional[str] = None,
):
    if name is None:
        name = id_util.UniqueStr("ImageTargetResize_")

    res_image, scale, new_size = image_resize(
        images,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        keep_aspect_ratio=True,
        resize_side=resize_side,
        interpolation_type=interpolation_type,
        name=name,
    )
    return res_image, new_size, scale


@oneflow_export("image.CropMirrorNormalize", "image.crop_mirror_normalize")
def CropMirrorNormalize(
    input_blob,
    mirror_blob=None,
    color_space="BGR",
    output_layout="NCHW",
    crop_h=0,
    crop_w=0,
    crop_pos_y=0.5,
    crop_pos_x=0.5,
    mean=[0.0],
    std=[1.0],
    output_dtype=dtype_util.float,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("CropMirrorNormalize_")
    op = (
        flow.user_op_builder(name).Op("crop_mirror_normalize").Input("in", [input_blob])
    )
    if mirror_blob is not None:
        op = op.Input("mirror", [mirror_blob])
    return (
        op.Output("out")
        .Attr("color_space", color_space)
        .Attr("output_layout", output_layout)
        .Attr("mean", mean)
        .Attr("std", std)
        .Attr("crop_h", crop_h)
        .Attr("crop_w", crop_w)
        .Attr("crop_pos_y", crop_pos_y)
        .Attr("crop_pos_x", crop_pos_x)
        .Attr("output_dtype", output_dtype)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("random.CoinFlip", "random.coin_flip")
def CoinFlip(batch_size=1, seed=None, probability=0.5, name=None):
    if name is None:
        name = id_util.UniqueStr("CoinFlip_")

    return (
        flow.user_op_builder(name)
        .Op("coin_flip")
        .Output("out")
        .Attr("batch_size", batch_size)
        .Attr("probability", probability)
        .SetRandomSeed(seed)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("image.decode", "image_decode")
def image_decode(
    images_bytes_buffer, dtype=dtype_util.uint8, color_space="BGR", name=None
):
    # TODO: check color_space valiad
    if name is None:
        name = id_util.UniqueStr("ImageDecode_")

    op = (
        flow.user_op_builder(name)
        .Op("image_decode")
        .Input("in", [images_bytes_buffer])
        .Output("out")
        .Attr("color_space", color_space)
        .Attr("data_type", dtype)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("image.batch_align", "image_batch_align")
def image_batch_align(images, shape, dtype, alignment, name=None):
    if name is None:
        name = id_util.UniqueStr("ImageBatchAlign_")

    op = (
        flow.user_op_builder(name)
        .Op("image_batch_align")
        .Input("in", [images])
        .Output("out")
        .Attr("shape", shape)
        .Attr("data_type", dtype)
        .Attr("alignment", alignment)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("image.normalize", "image_normalize")
def image_normalize(image, std, mean, name=None):
    if name is None:
        name = id_util.UniqueStr("ImageNormalize_")

    assert isinstance(std, (list, tuple))
    assert isinstance(mean, (list, tuple))

    op = (
        flow.user_op_builder(name)
        .Op("image_normalize")
        .Input("in", [image])
        .Output("out")
        .Attr("std", std)
        .Attr("mean", mean)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("image.flip", "image_flip")
def image_flip(image, flip_code, name=None):
    assert isinstance(image, BlobDef)

    if name is None:
        name = id_util.UniqueStr("ImageFlip_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code,
            shape=(image.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode_".format(name),
        )
    else:
        assert image.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("image_flip")
        .Input("in", [image])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("detection.object_bbox_flip", "object_bbox_flip")
def object_bbox_flip(bbox, image_size, flip_code, name=None):
    assert isinstance(bbox, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert bbox.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectBboxFlip_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code,
            shape=(bbox.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode".format(name),
        )
    else:
        assert bbox.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("object_bbox_flip")
        .Input("bbox", [bbox])
        .Input("image_size", [image_size])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("detection.object_bbox_scale", "object_bbox_scale")
def object_bbox_scale(bbox, scale, name=None):
    assert isinstance(bbox, BlobDef)
    assert isinstance(scale, BlobDef)
    assert bbox.shape[0] == scale.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectBboxScale_")

    op = (
        flow.user_op_builder(name)
        .Op("object_bbox_scale")
        .Input("bbox", [bbox])
        .Input("scale", [scale])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export(
    "detection.object_segmentation_polygon_flip", "object_segmentation_polygon_flip"
)
def object_segm_poly_flip(poly, image_size, flip_code, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert poly.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyFilp_")

    if not isinstance(flip_code, BlobDef):
        assert isinstance(flip_code, int)
        flip_code = flow.constant(
            flip_code,
            shape=(poly.shape[0],),
            dtype=flow.int8,
            name="{}_FlipCode".format(name),
        )
    else:
        assert poly.shape[0] == flip_code.shape[0]

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_flip")
        .Input("poly", [poly])
        .Input("image_size", [image_size])
        .Input("flip_code", [flip_code])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export(
    "detection.object_segmentation_polygon_scale", "object_segmentation_polygon_scale"
)
def object_segm_poly_scale(poly, scale, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(scale, BlobDef)
    assert poly.shape[0] == scale.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyFilp_")

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_scale")
        .Input("poly", [poly])
        .Input("scale", [scale])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export(
    "detection.object_segmentation_polygon_to_mask",
    "object_segmentation_polygon_to_mask",
)
def object_segm_poly_to_mask(poly, poly_index, image_size, name=None):
    assert isinstance(poly, BlobDef)
    assert isinstance(poly_index, BlobDef)
    assert isinstance(image_size, BlobDef)
    assert poly.shape[0] == poly_index.shape[0]
    assert poly.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyToMask_")

    op = (
        flow.user_op_builder(name)
        .Op("object_segmentation_polygon_to_mask")
        .Input("poly", [poly])
        .Input("poly_index", [poly_index])
        .Input("image_size", [image_size])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("data.coco_reader")
def api_coco_reader(
    annotation_file,
    image_dir,
    batch_size,
    shuffle=True,
    random_seed=None,
    group_by_aspect_ratio=True,
    stride_partition=True,
    name=None,
):
    import random
    import sys

    if name is None:
        name = id_util.UniqueStr("COCOReader_")

    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)

    op = (
        flow.consistent_user_op_builder(name)
        .Op("COCOReader")
        .Output("image")
        .Output("image_id")
        .Output("image_size")
        .Output("gt_bbox")
        .Output("gt_label")
        .Output("gt_segm")
        .Output("gt_segm_index")
        .Attr("annotation_file", annotation_file)
        .Attr("image_dir", image_dir)
        .Attr("batch_size", batch_size)
        .Attr("shuffle_after_epoch", shuffle)
        .Attr("random_seed", random_seed)
        .Attr("group_by_ratio", group_by_aspect_ratio)
        .Attr("stride_partition", stride_partition)
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()
