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
import oneflow.python.framework.module as module_util

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.remote_blob import BlobDef
from typing import Optional, Sequence, Union
import random
import sys
import numpy


@oneflow_export("data.OFRecordRawDecoder", "data.ofrecord_raw_decoder")
def OFRecordRawDecoder(
    input_blob: BlobDef,
    blob_name: str,
    shape: Sequence[int],
    dtype: dtype_util.dtype,
    dim1_varying_length: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
) -> BlobDef:
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
def api_ofrecord_image_decoder_random_crop(
    input_blob: BlobDef,
    blob_name: str,
    color_space: str = "BGR",
    num_attempts: int = 10,
    seed: Optional[int] = None,
    random_area: Sequence[float] = [0.08, 1.0],
    random_aspect_ratio: Sequence[float] = [0.75, 1.333333],
    name: str = "OFRecordImageDecoderRandomCrop",
) -> BlobDef:
    assert isinstance(name, str)
    if seed is not None:
        assert name is not None
    module = flow.find_or_create_module(
        name,
        lambda: OFRecordImageDecoderRandomCropModule(
            blob_name=blob_name,
            color_space=color_space,
            num_attempts=num_attempts,
            random_seed=seed,
            random_area=random_area,
            random_aspect_ratio=random_aspect_ratio,
            name=name,
        ),
    )
    return module(input_blob)


class OFRecordImageDecoderRandomCropModule(module_util.Module):
    def __init__(
        self,
        blob_name: str,
        color_space: str,
        num_attempts: int,
        random_seed: Optional[int],
        random_area: Sequence[float],
        random_aspect_ratio: Sequence[float],
        name: str,
    ):
        module_util.Module.__init__(self, name)
        seed, has_seed = flow.random.gen_seed(random_seed)
        self.op_module_builder = (
            flow.user_op_module_builder("ofrecord_image_decoder_random_crop")
            .InputSize("in", 1)
            .Output("out")
            .Attr("name", blob_name)
            .Attr("color_space", color_space)
            .Attr("num_attempts", num_attempts)
            .Attr("random_area", random_area)
            .Attr("random_aspect_ratio", random_aspect_ratio)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self, input: BlobDef):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("OFRecordImageDecoderRandomCrop_")

        return (
            self.op_module_builder.OpName(name)
            .Input("in", [input])
            .Build()
            .InferAndTryRun()
            .SoleOutputBlob()
        )


@oneflow_export("data.OFRecordImageDecoder", "data.ofrecord_image_decoder")
def OFRecordImageDecoder(
    input_blob: BlobDef,
    blob_name: str,
    color_space: str = "BGR",
    name: Optional[str] = None,
) -> BlobDef:
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
def Resize(
    input_blob: BlobDef,
    color_space: str = "BGR",
    interp_type: str = "Linear",
    resize_shorter: int = 0,
    resize_x: int = 0,
    resize_y: int = 0,
    name: Optional[str] = None,
) -> BlobDef:
    if name is None:
        name = id_util.UniqueStr("ImageResize_")
    return (
        flow.user_op_builder(name)
        .Op("image_resize")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("color_space", color_space)
        .Attr("interp_type", interp_type)
        .Attr("resize_shorter", resize_shorter)
        .Attr("resize_x", resize_x)
        .Attr("resize_y", resize_y)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("image.CropMirrorNormalize", "image.crop_mirror_normalize")
def CropMirrorNormalize(
    input_blob: BlobDef,
    mirror_blob: Optional[BlobDef] = None,
    color_space: str = "BGR",
    output_layout: str = "NCHW",
    crop_h: int = 0,
    crop_w: int = 0,
    crop_pos_y: float = 0.5,
    crop_pos_x: float = 0.5,
    mean: Sequence[float] = [0.0],
    std: Sequence[float] = [1.0],
    output_dtype: dtype_util.dtype = dtype_util.float,
    name: Optional[str] = None,
):
    if name is None:
        name = id_util.UniqueStr("CropMirrorNormalize_")
    op_type_name = ""
    if input_blob.dtype is dtype_util.tensor_buffer:
        op_type_name = "crop_mirror_normalize_from_tensorbuffer"
    elif input_blob.dtype is dtype_util.uint8:
        op_type_name = "crop_mirror_normalize_from_uint8"
    else:
        print(
            "ERROR! oneflow.data.crop_mirror_normalize op",
            " NOT support input data type : ",
            input_blob.dtype,
        )
        raise NotImplementedError

    op = flow.user_op_builder(name).Op(op_type_name).Input("in", [input_blob])
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


@oneflow_export("image.random_crop", "image_random_crop")
def api_image_random_crop(
    input_blob: BlobDef,
    num_attempts: int = 10,
    seed: Optional[int] = None,
    random_area: Sequence[float] = None,
    random_aspect_ratio: Sequence[float] = None,
    name: str = "ImageRandomCrop",
) -> BlobDef:
    assert isinstance(name, str)
    if seed is not None:
        assert name is not None
    if random_area is None:
        random_area = [0.08, 1.0]
    if random_aspect_ratio is None:
        random_aspect_ratio = [0.75, 1.333333]
    module = flow.find_or_create_module(
        name,
        lambda: ImageRandomCropModule(
            num_attempts=num_attempts,
            random_seed=seed,
            random_area=random_area,
            random_aspect_ratio=random_aspect_ratio,
            name=name,
        ),
    )
    return module(input_blob)


class ImageRandomCropModule(module_util.Module):
    def __init__(
        self,
        num_attempts: int,
        random_seed: Optional[int],
        random_area: Sequence[float],
        random_aspect_ratio: Sequence[float],
        name: str,
    ):
        module_util.Module.__init__(self, name)
        seed, has_seed = flow.random.gen_seed(random_seed)
        self.op_module_builder = (
            flow.user_op_module_builder("image_random_crop")
            .InputSize("in", 1)
            .Output("out")
            .Attr("num_attempts", num_attempts)
            .Attr("random_area", random_area)
            .Attr("random_aspect_ratio", random_aspect_ratio)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self, input: BlobDef):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("ImageRandomCrop_")

        return (
            self.op_module_builder.OpName(name)
            .Input("in", [input])
            .Build()
            .InferAndTryRun()
            .SoleOutputBlob()
        )


@oneflow_export("random.CoinFlip", "random.coin_flip")
def api_coin_flip(
    batch_size: int = 1,
    seed: Optional[int] = None,
    probability: float = 0.5,
    name: str = "CoinFlip",
) -> BlobDef:
    assert isinstance(name, str)
    if seed is not None:
        assert name is not None
    module = flow.find_or_create_module(
        name,
        lambda: CoinFlipModule(
            batch_size=batch_size, probability=probability, random_seed=seed, name=name,
        ),
    )
    return module()


class CoinFlipModule(module_util.Module):
    def __init__(
        self,
        batch_size: str,
        probability: float,
        random_seed: Optional[int],
        name: str,
    ):
        module_util.Module.__init__(self, name)
        seed, has_seed = flow.random.gen_seed(random_seed)
        self.op_module_builder = (
            flow.user_op_module_builder("coin_flip")
            .Output("out")
            .Attr("batch_size", batch_size)
            .Attr("probability", probability)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("CoinFlip_")

        return (
            self.op_module_builder.OpName(name)
            .Build()
            .InferAndTryRun()
            .SoleOutputBlob()
        )


@oneflow_export("image.decode", "image_decode")
def image_decode(
    images_bytes_buffer: BlobDef,
    dtype: dtype_util.dtype = dtype_util.uint8,
    color_space: str = "BGR",
    name: Optional[str] = None,
) -> BlobDef:
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


@oneflow_export("image.target_resize", "image_target_resize")
def image_target_resize(
    images: BlobDef, target_size: int, max_size: int, name: Optional[str] = None
) -> Sequence[BlobDef]:
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
        .Attr("target_size", target_size)
        .Attr("max_size", max_size)
        .Build()
    )
    return op.InferAndTryRun().RemoteBlobList()


@oneflow_export("image.batch_align", "image_batch_align")
def image_batch_align(
    images: BlobDef,
    shape: Sequence[int],
    dtype: dtype_util.dtype,
    alignment: int,
    name: Optional[str] = None,
) -> BlobDef:
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
def image_normalize(
    image: BlobDef,
    std: Sequence[float],
    mean: Sequence[float],
    name: Optional[str] = None,
) -> BlobDef:
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
def image_flip(
    image: BlobDef, flip_code: Union[int, BlobDef], name: Optional[str] = None
) -> BlobDef:
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
def object_bbox_flip(
    bbox: BlobDef,
    image_size: BlobDef,
    flip_code: Union[int, BlobDef],
    name: Optional[str] = None,
) -> BlobDef:
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
def object_bbox_scale(
    bbox: BlobDef, scale: BlobDef, name: Optional[str] = None
) -> BlobDef:
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
def object_segm_poly_flip(
    poly: BlobDef,
    image_size: BlobDef,
    flip_code: Union[int, BlobDef],
    name: Optional[str] = None,
) -> BlobDef:
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
def object_segm_poly_scale(
    poly: BlobDef, scale: BlobDef, name: Optional[str] = None
) -> BlobDef:
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
def object_segm_poly_to_mask(
    poly: BlobDef, poly_index: BlobDef, image_size: BlobDef, name: Optional[str] = None
) -> BlobDef:
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
    annotation_file: str,
    image_dir: str,
    batch_size: int,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
    group_by_aspect_ratio: bool = True,
    stride_partition: bool = True,
    name: str = None,
) -> BlobDef:
    assert name is not None
    module = flow.find_or_create_module(
        name,
        lambda: COCOReader(
            annotation_file=annotation_file,
            image_dir=image_dir,
            batch_size=batch_size,
            shuffle=shuffle,
            random_seed=random_seed,
            group_by_aspect_ratio=group_by_aspect_ratio,
            stride_partition=stride_partition,
            name=name,
        ),
    )
    return module()


class COCOReader(module_util.Module):
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        batch_size: int,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
        group_by_aspect_ratio: bool = True,
        stride_partition: bool = True,
        name: str = None,
    ):
        assert name is not None
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)
        module_util.Module.__init__(self, name)
        self.op_module_builder = (
            flow.consistent_user_op_module_builder("COCOReader")
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
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("COCOReader")
        return (
            self.op_module_builder.OpName(name)
            .Build()
            .InferAndTryRun()
            .RemoteBlobList()
        )


@oneflow_export("data.ofrecord_image_classification_reader")
def ofrecord_image_classification_reader(
    ofrecord_dir: str,
    image_feature_name: str,
    label_feature_name: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    shuffle_after_epoch: bool = False,
    color_space: str = "BGR",
    decode_buffer_size_per_thread: int = 32,
    num_decode_threads_per_machine: Optional[int] = None,
    name: Optional[str] = None,
) -> BlobDef:
    if name is None:
        name = id_util.UniqueStr("OFRecordImageClassificationReader_")
    (image, label) = (
        flow.user_op_builder(name)
        .Op("ofrecord_image_classification_reader")
        .Output("image")
        .Output("label")
        .Attr("data_dir", ofrecord_dir)
        .Attr("data_part_num", data_part_num)
        .Attr("batch_size", batch_size)
        .Attr("part_name_prefix", part_name_prefix)
        .Attr("random_shuffle", random_shuffle)
        .Attr("shuffle_buffer_size", shuffle_buffer_size)
        .Attr("shuffle_after_epoch", shuffle_after_epoch)
        .Attr("part_name_suffix_length", part_name_suffix_length)
        .Attr("color_space", color_space)
        .Attr("image_feature_name", image_feature_name)
        .Attr("label_feature_name", label_feature_name)
        .Attr("decode_buffer_size_per_thread", decode_buffer_size_per_thread)
        .Attr("num_decode_threads_per_machine", num_decode_threads_per_machine or 0)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    label = flow.tensor_buffer_to_tensor(label, dtype=flow.int32, instance_shape=[1])
    label = flow.squeeze(label, axis=[-1])
    return image, label


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

    op = flow.user_op_builder(name).Op("ssd_random_crop")
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
