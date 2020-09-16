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
import traceback


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


@oneflow_export("image.Resize", "image.resize", "image_resize")
def api_image_resize(
    image: BlobDef,
    target_size: Union[int, Sequence[int]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    keep_aspect_ratio: bool = False,
    resize_side: str = "shorter",
    channels: int = 3,
    dtype: Optional[dtype_util.dtype] = None,
    interpolation_type: str = "auto",
    name: Optional[str] = None,
    # deprecated params, reserve for backward compatible
    color_space: Optional[str] = None,
    interp_type: Optional[str] = None,
    resize_shorter: int = 0,
    resize_x: int = 0,
    resize_y: int = 0,
) -> Union[BlobDef, Sequence[BlobDef]]:
    r"""Resize images to target size.

    Args:
        image: A `Tensor` consists of images to be resized.
        target_size: A list or tuple when `keep_aspect_ratio` is false or an int when
            `keep_aspect_ratio` is true. When `keep_aspect_ratio` is false, `target_size` has
            a form of `(target_width, target_height)` that image will resize to. When
            `keep_aspect_ratio` is true, the longer side or shorter side of the image
            will be resized to target size.
        min_size: An int, optional. Only works when `keep_aspect_ratio` is true and `resize_side`
            is "longer". If `min_size` is not None, the shorter side must be greater than or
            equal to `min_size`. Default is None.
        max_size: An int, optional. Only works when `keep_aspect_ratio` is true and `resize_side`
            is "shorter". If `max_size` is not None, the longer side must be less than or equal
            to `max_size`. Default is None.
        keep_aspect_ratio: A bool. If is false, indicate that image will be resized to fixed
            width and height, otherwise image will be resized keeping aspect ratio.
        resize_side: A str of "longer" or "shorter". Only works when `keep_aspect_ratio` is True.
            If `resize_side` is "longer", the longer side of image will be resized to `target_size`.
            If `resize_side` is "shorter", the shorter side of image will be resized to
            `target_size`.
        channels: An int. how many channels an image has
        dtype: `oneflow.dtype`. Indicate output resized image data type.
        interpolation_type: A str of "auto", "bilinear", "nearest_neighbor", "bicubic" or "area".
            Indicate interpolation method used to resize image.
        name: A str, optional. Name for the operation.
        color_space: Deprecated, a str of "RGB", "BGR" or "GRAY". Please use `channels` instead.
        interp_type: Deprecated, s str of "Linear", "Cubic" or "NN". Please use `interpolation_type`
            instead.
        resize_shorter: Deprecated, a int. Indicate target size that the shorter side of image will
            resize to. Please use `target_size` and `resize_side` instead.
        resize_x: Deprecated, a int. Indicate the target size that the width of image will resize to.
            Please use `target_size` instead.
        resize_y: Deprecated, a int. Indicate the target size that the height of image will resize to.
            Please use `target_size` instead.

    Returns:
        Tuple of resized images `Blob`, width and height scales `Blob` and new width and height `Blob`
        (new width and height `Blob` will be None when keep_aspect_ratio is false).
        If deprecated params are used, a single resized images `Blob` will be returned.
    """
    # process deprecated params
    deprecated_param_used = False
    if color_space is not None:
        print("WARNING: color_space has been deprecated. Please use channels instead.")
        print(traceback.format_stack()[-2])
        deprecated_param_used = True
        assert isinstance(color_space, str)
        if color_space.upper() == "RGB" or color_space.upper() == "BGR":
            channels = 3
        elif color_space.upper() == "GRAY":
            channels = 1
        else:
            raise ValueError("invalid color_space")

    if interp_type is not None:
        print(
            "WARNING: interp_type has been deprecated. Please use interpolation_type instead."
        )
        print(traceback.format_stack()[-2])
        deprecated_param_used = True
        assert isinstance(interp_type, str)
        if interp_type == "Linear":
            interpolation_type = "bilinear"
        elif interp_type == "NN":
            interpolation_type = "nearest_neighbor"
        elif interp_type == "Cubic":
            interpolation_type = "bicubic"
        else:
            raise ValueError("invalid interp_type")

    if resize_x > 0 and resize_y > 0:
        print(
            "WARNING: resize_x and resize_y has been deprecated. Please use target_size instead."
        )
        print(traceback.format_stack()[-2])
        deprecated_param_used = True
        target_size = (resize_x, resize_y)
        keep_aspect_ratio = False

    if resize_shorter > 0:
        print(
            "WARNING: resize_shorter has been deprecated. Please use target_size instead."
        )
        print(traceback.format_stack()[-2])
        deprecated_param_used = True
        target_size = resize_shorter
        keep_aspect_ratio = True
        resize_side = "shorter"

    if name is None:
        name = id_util.UniqueStr("ImageResize_")

    if keep_aspect_ratio:
        if not isinstance(target_size, int):
            raise ValueError(
                "target_size must be an int when keep_aspect_ratio is True"
            )

        if min_size is None:
            min_size = 0

        if max_size is None:
            max_size = 0

        if resize_side == "shorter":
            resize_longer = False
        elif resize_side == "longer":
            resize_longer = True
        else:
            raise ValueError('resize_side must be "shorter" or "longer"')

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
        scale = flow.tensor_buffer_to_tensor(
            scale, dtype=flow.float32, instance_shape=(2,)
        )
        new_size = flow.tensor_buffer_to_tensor(
            new_size, dtype=flow.int32, instance_shape=(2,)
        )

    else:
        if (
            not isinstance(target_size, (list, tuple))
            or len(target_size) != 2
            or not all(isinstance(size, int) for size in target_size)
        ):
            raise ValueError(
                "target_size must be a form like (width, height) when keep_aspect_ratio is False"
            )

        if dtype is None:
            dtype = flow.uint8

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

    if deprecated_param_used:
        return res_image

    return res_image, scale, new_size


@oneflow_export("image.target_resize", "image_target_resize")
def api_image_target_resize(
    images: BlobDef,
    target_size: int,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    resize_side: str = "shorter",
    interpolation_type: str = "auto",
    name: Optional[str] = None,
) -> Sequence[BlobDef]:
    if name is None:
        name = id_util.UniqueStr("ImageTargetResize_")

    res_image, scale, new_size = api_image_resize(
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
) -> BlobDef:
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


@oneflow_export("data.TFRecordRawDecoder")
def TFRecordRawDecoder(
    input_blob: BlobDef,
    blob_name: str,
    shape: Sequence[int],
    dtype: dtype_util.dtype,
    dim1_varying_length: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
) -> BlobDef:
    if name is None:
        name = id_util.UniqueStr("TFRecordRawDecoder_")
    return (
        flow.user_op_builder(name)
        .Op("tfrecord_raw_decoder")
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


@oneflow_export("data.TFRecordImageDecoderRandomCrop")
def TFRecordImageDecoderRandomCrop(
    input_blob: BlobDef,
    blob_name: str,
    color_space: str = "BGR",
    num_attempts: int = 10,
    seed: Optional[int] = None,
    random_area: Sequence[float] = [0.08, 1.0],
    random_aspect_ratio: Sequence[float] = [0.75, 1.333333],
    name: str = "TFRecordImageDecoderRandomCrop",
) -> BlobDef:
    assert isinstance(name, str)
    if seed is not None:
        assert name is not None
    module = flow.find_or_create_module(
        name,
        lambda: TFRecordImageDecoderRandomCropModule(
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


class TFRecordImageDecoderRandomCropModule(module_util.Module):
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
            flow.user_op_module_builder("tfrecord_image_decoder_random_crop")
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
            name = id_util.UniqueStr("TFRecordImageDecoderRandomCrop_")

        return (
            self.op_module_builder.OpName(name)
            .Input("in", [input])
            .Build()
            .InferAndTryRun()
            .SoleOutputBlob()
        )


@oneflow_export("data.TFRecordImageDecoder")
def TFRecordImageDecoder(
    input_blob: BlobDef,
    blob_name: str,
    color_space: str = "BGR",
    name: Optional[str] = None,
) -> BlobDef:
    if name is None:
        name = id_util.UniqueStr("TFRecordImageDecoder_")
    return (
        flow.user_op_builder(name)
        .Op("tfrecord_image_decoder")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("name", blob_name)
        .Attr("color_space", color_space)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
