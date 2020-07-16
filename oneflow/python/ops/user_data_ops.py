from __future__ import absolute_import

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import numpy

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.remote_blob import BlobDef
from typing import Sequence, Optional


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
        .Attr("name", blob_name, "AttrTypeString")
        .Attr("shape", shape, "AttrTypeShape")
        .Attr("data_type", dtype, "AttrTypeInt64")
        .Attr("dim1_varying_length", dim1_varying_length, "AttrTypeBool")
        .Attr("auto_zero_padding", auto_zero_padding, "AttrTypeBool")
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
        .Attr("name", blob_name, "AttrTypeString")
        .Attr("color_space", color_space, "AttrTypeString")
        .Attr("num_attempts", num_attempts, "AttrTypeInt32")
        .SetRandomSeed(seed)
        .Attr("random_area", random_area, "AttrTypeListFloat")
        .Attr("random_aspect_ratio", random_aspect_ratio, "AttrTypeListFloat")
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
        .Attr("name", blob_name, "AttrTypeString")
        .Attr("color_space", color_space, "AttrTypeString")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("image.Resize", "image.resize")
def Resize(
    input_blob,
    color_space="BGR",
    interp_type="Linear",
    resize_shorter=0,
    resize_x=0,
    resize_y=0,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("ImageResize_")
    return (
        flow.user_op_builder(name)
        .Op("image_resize")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("color_space", color_space, "AttrTypeString")
        .Attr("interp_type", interp_type, "AttrTypeString")
        .Attr("resize_shorter", resize_shorter, "AttrTypeInt64")
        .Attr("resize_x", resize_x, "AttrTypeInt64")
        .Attr("resize_y", resize_y, "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


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
    output_dtype=flow.float,
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
        .Attr("color_space", color_space, "AttrTypeString")
        .Attr("output_layout", output_layout, "AttrTypeString")
        .Attr("mean", mean, "AttrTypeListFloat")
        .Attr("std", std, "AttrTypeListFloat")
        .Attr("crop_h", crop_h, "AttrTypeInt64")
        .Attr("crop_w", crop_w, "AttrTypeInt64")
        .Attr("crop_pos_y", crop_pos_y, "AttrTypeFloat")
        .Attr("crop_pos_x", crop_pos_x, "AttrTypeFloat")
        .Attr("output_dtype", output_dtype, "AttrTypeInt32")
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
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("probability", probability, "AttrTypeFloat")
        .SetRandomSeed(seed)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("image.decode", "image_decode")
def image_decode(images_bytes_buffer, dtype=flow.uint8, color_space="BGR", name=None):
    # TODO: check color_space valiad
    if name is None:
        name = id_util.UniqueStr("ImageDecode_")

    op = (
        flow.user_op_builder(name)
        .Op("image_decode")
        .Input("in", [images_bytes_buffer])
        .Output("out")
        .Attr("color_space", color_space, "AttrTypeString")
        .Attr("data_type", dtype, "AttrTypeDataType")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("image.target_resize", "image_target_resize")
def image_target_resize(images, target_size, max_size, name=None):
    # TODO: check target_size and max_size valid
    if name is None:
        name = id_util.UniqueStr("ImageTargetResize_")

    op = (
        flow.user_op_builder(name)
        .Op("image_target_resize")
        .Input("in", [images])
        .Output("out")
        .Output("size")
        .Output("scale")
        .Attr("target_size", target_size, "AttrTypeInt32")
        .Attr("max_size", max_size, "AttrTypeInt32")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()


@oneflow_export("image.batch_align", "image_batch_align")
def image_batch_align(images, shape, dtype, alignment, name=None):
    if name is None:
        name = id_util.UniqueStr("ImageBatchAlign_")

    op = (
        flow.user_op_builder(name)
        .Op("image_batch_align")
        .Input("in", [images])
        .Output("out")
        .Attr("shape", shape, "AttrTypeShape")
        .Attr("data_type", dtype, "AttrTypeDataType")
        .Attr("alignment", alignment, "AttrTypeInt32")
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
        .Attr("std", std, "AttrTypeListFloat")
        .Attr("mean", mean, "AttrTypeListFloat")
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
        .Attr("annotation_file", annotation_file, "AttrTypeString")
        .Attr("image_dir", image_dir, "AttrTypeString")
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("shuffle_after_epoch", shuffle, "AttrTypeBool")
        .Attr("random_seed", random_seed, "AttrTypeInt64")
        .Attr("group_by_ratio", group_by_aspect_ratio, "AttrTypeBool")
        .Attr("stride_partition", stride_partition, "AttrTypeBool")
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()


@oneflow_export("ssd_random_crop")
def ssd_random_crop(
    image: BlobDef,
    bbox: Optional[BlobDef] = None,
    label: Optional[BlobDef] = None,
    iou_overlap_ranges: Optional[Sequence[Optional[Sequence[Optional[float]]]]] = None,
    size_shrink_rates: Optional[Sequence[Sequence[float]]] = None,
    aspect_ratio_range: Optional[Sequence[float]] = None,
    random_seed: Optional[int] = None,
    max_num_attempts: Optional[int] = None,
    name: Optional[str] = None,
):
    if name is None:
        name = id_util.UniqueStr("SSDRandomCrop_")

    op = flow.consistent_user_op_builder(name).Op("ssd_random_crop")
    op.Input("image", [image]).Output("out_image")

    if bbox is not None:
        op.Input("bbox", [bbox]).Output("out_bbox")

    if label is not None:
        op.Input("label", [label]).Output("out_label")

    if iou_overlap_ranges is not None:
        min_ious = []
        max_ious = []
        assert len(iou_overlap_ranges) > 0
        for iou_range in iou_overlap_ranges:
            if iou_range is None:
                min_ious.append(-1.0)
                max_ious.append(-1.0)
            else:
                assert len(iou_range) == 2
                min_iou = 0.0
                max_iou = 1.0
                if iou_range[0] is not None:
                    min_iou = iou_range[0]
                if iou_range[1] is not None:
                    max_iou = iou_range[1]
                assert 0.0 <= min_iou <= 1.0
                assert 0.0 <= max_iou <= 1.0
                assert min_iou < max_iou
                min_ious.append(min_iou)
                max_ious.append(max_iou)

        op.Attr("min_iou_overlaps", min_ious).Attr("max_iou_overlaps", max_ious)

    if size_shrink_rates is not None:
        shrink_rate_array = numpy.array(size_shrink_rates)
        assert shrink_rate_array.shape == (2, 2)
        assert numpy.all(shrink_rate_array[0, :] > 0.0)
        assert numpy.all(shrink_rate_array[1, :] <= 1.0)
        assert numpy.all(shrink_rate_array[0, :] <= shrink_rate_array[1, :])
        op.Attr("min_width_shrink_rate", shrink_rate_array[0][0]).Attr(
            "max_width_shrink_rate", shrink_rate_array[0][1]
        ).Attr("min_height_shrink_rate", shrink_rate_array[1][0]).Attr(
            "max_height_shrink_rate", shrink_rate_array[1][1]
        )

    if aspect_ratio_range is not None:
        assert len(aspect_ratio_range) == 2
        assert all(a > 0.0 for a in aspect_ratio_range)
        assert aspect_ratio_range[0] < aspect_ratio_range[1]
        op.Attr("min_crop_aspect_ratio", aspect_ratio_range[0]).Attr(
            "max_crop_aspect_ratio", aspect_ratio_range[1]
        )

    if random_seed is not None:
        op.Attr("has_seed", True).Attr("seed", random_seed)

    if max_num_attempts is not None:
        op.Attr("max_num_attempts", max_num_attempts)

    return op.Build().InferAndTryRun().RemoteBlobList()
