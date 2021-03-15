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
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.module as module_util
import oneflow_api
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Sequence, Union
import random
import sys
import traceback


@oneflow_export("data.OFRecordRawDecoder", "data.ofrecord_raw_decoder")
def OFRecordRawDecoder(
    input_blob: oneflow_api.BlobDesc,
    blob_name: str,
    shape: Sequence[int],
    dtype: flow.dtype,
    dim1_varying_length: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
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


@oneflow_export("data.OFRecordBytesDecoder", "data.ofrecord_bytes_decoder")
def OFRecordBytesDecoder(
    input_blob: oneflow_api.BlobDesc, blob_name: str, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    if name is None:
        name = id_util.UniqueStr("OFRecordBytesDecoder_")
    return (
        flow.user_op_builder(name)
        .Op("ofrecord_bytes_decoder")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("name", blob_name)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export(
    "data.OFRecordImageDecoderRandomCrop", "data.ofrecord_image_decoder_random_crop"
)
def api_ofrecord_image_decoder_random_crop(
    input_blob: oneflow_api.BlobDesc,
    blob_name: str,
    color_space: str = "BGR",
    num_attempts: int = 10,
    seed: Optional[int] = None,
    random_area: Sequence[float] = [0.08, 1.0],
    random_aspect_ratio: Sequence[float] = [0.75, 1.333333],
    name: str = "OFRecordImageDecoderRandomCrop",
) -> oneflow_api.BlobDesc:
    """This operator is an image decoder with random crop. 

    Args:
        input_blob (oneflow_api.BlobDesc): The input Blob
        blob_name (str): The name of the Blob
        color_space (str, optional): The color space, such as "RGB", "BGR". Defaults to "BGR".
        num_attempts (int, optional): The maximum number of random cropping attempts. Defaults to 10.
        seed (Optional[int], optional): The random seed. Defaults to None.
        random_area (Sequence[float], optional): The random cropping area. Defaults to [0.08, 1.0].
        random_aspect_ratio (Sequence[float], optional): The random scaled ratio. Defaults to [0.75, 1.333333].
        name (str, optional): The name for the operation. Defaults to "OFRecordImageDecoderRandomCrop".

    Returns:
        oneflow_api.BlobDesc: The random cropped Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def ofrecord_reader_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 16
            color_space = "RGB"
            # our ofrecord file path is "./dataset/part-0"
            ofrecord = flow.data.ofrecord_reader(
                "./imgdataset",
                batch_size=batch_size,
                data_part_num=1,
                part_name_suffix_length=-1,
                part_name_prefix='part-', 
                random_shuffle=True,
                shuffle_after_epoch=True,
            )
            image = flow.data.OFRecordImageDecoderRandomCrop(
                    ofrecord, "encoded", color_space=color_space
                )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(224, 224)
                )
            label = flow.data.OFRecordRawDecoder(
                ofrecord, "class/label", shape=(1, ), dtype=flow.int32
            )

            return res_image, label

        if __name__ == "__main__":
            images, labels = ofrecord_reader_job()
            # images.shape (16, 224, 224, 3)

    """
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

    def forward(self, input: oneflow_api.BlobDesc):
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
    input_blob: oneflow_api.BlobDesc,
    blob_name: str,
    color_space: str = "BGR",
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator is an image decoder. 

    Args:
        input_blob (oneflow_api.BlobDesc): The input Blob
        blob_name (str): The name of the input Blob
        color_space (str, optional): The color space, such as "RGB", "BGR". Defaults to "BGR".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def image_decoder_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 16
            color_space = "RGB"
            # our ofrecord file path is "./dataset/part-0"
            ofrecord = flow.data.ofrecord_reader(
                "./imgdataset",
                batch_size=batch_size,
                data_part_num=1,
                part_name_suffix_length=-1,
                part_name_prefix='part-', 
                random_shuffle=True,
                shuffle_after_epoch=True,
            )
            image = flow.data.OFRecordImageDecoder(
                    ofrecord, "encoded", color_space=color_space
                )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(224, 224)
                )
            label = flow.data.OFRecordRawDecoder(
                ofrecord, "class/label", shape=(1, ), dtype=flow.int32
            )

            return res_image, label

        if __name__ == "__main__":
            images, labels = image_decoder_job()
            # image.shape (16, 224, 224, 3)

    """
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
    image: oneflow_api.BlobDesc,
    target_size: Union[int, Sequence[int]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    keep_aspect_ratio: bool = False,
    resize_side: str = "shorter",
    channels: int = 3,
    dtype: Optional[flow.dtype] = None,
    interpolation_type: str = "auto",
    name: Optional[str] = None,
    # deprecated params, reserve for backward compatible
    color_space: Optional[str] = None,
    interp_type: Optional[str] = None,
    resize_shorter: int = 0,
    resize_x: int = 0,
    resize_y: int = 0,
) -> Union[oneflow_api.BlobDesc, Sequence[oneflow_api.BlobDesc]]:
    r"""Resize images to target size.

    Args:
        image: A `Tensor` consists of images to be resized.
        target_size: A list or tuple when `keep_aspect_ratio` is false or an int when `keep_aspect_ratio` is true. When `keep_aspect_ratio` is false, `target_size` has a form of `(target_width, target_height)` that image will resize to. When `keep_aspect_ratio` is true, the longer side or shorter side of the image will be resized to target size.
        min_size: An int, optional. Only works when `keep_aspect_ratio` is true and `resize_side` is "longer". If `min_size` is not None, the shorter side must be greater than or equal to `min_size`. Default is None.
        max_size: An int, optional. Only works when `keep_aspect_ratio` is true and `resize_side` is "shorter". If `max_size` is not None, the longer side must be less than or equal to `max_size`. Default is None.
        keep_aspect_ratio: A bool. If is false, indicate that image will be resized to fixed width and height, otherwise image will be resized keeping aspect ratio.
        resize_side: A str of "longer" or "shorter". Only works when `keep_aspect_ratio` is True. If `resize_side` is "longer", the longer side of image will be resized to `target_size`. If `resize_side` is "shorter", the shorter side of image will be resized to `target_size`.
        channels: An int. how many channels an image has
        dtype: `oneflow.dtype`. Indicate output resized image data type.
        interpolation_type: A str of "auto", "bilinear", "nearest_neighbor", "bicubic" or "area". Indicate interpolation method used to resize image.
        name: A str, optional. Name for the operation.
        color_space: Deprecated, a str of "RGB", "BGR" or "GRAY". Please use `channels` instead.
        interp_type: Deprecated, s str of "Linear", "Cubic" or "NN". Please use `interpolation_type` instead.
        resize_shorter: Deprecated, a int. Indicate target size that the shorter side of image will resize to. Please use `target_size` and `resize_side` instead.
        resize_x: Deprecated, a int. Indicate the target size that the width of image will resize to. Please use `target_size` instead.
        resize_y: Deprecated, a int. Indicate the target size that the height of image will resize to. Please use `target_size` instead.

    Returns:
        Tuple of resized images `Blob`, width and height scales `Blob` and new width and height `Blob`
        (new width and height `Blob` will be None when keep_aspect_ratio is false).
        If deprecated params are used, a single resized images `Blob` will be returned.

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def ofrecord_reader_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 16
            color_space = "RGB"
            # our ofrecord file path is "./dataset/part-0"
            ofrecord = flow.data.ofrecord_reader(
                "./imgdataset",
                batch_size=batch_size,
                data_part_num=1,
                part_name_suffix_length=-1,
                part_name_prefix='part-', 
                random_shuffle=True,
                shuffle_after_epoch=True,
            )
            image = flow.data.OFRecordImageDecoderRandomCrop(
                    ofrecord, "encoded", color_space=color_space
                )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(224, 224)
                )
            label = flow.data.OFRecordRawDecoder(
                ofrecord, "class/label", shape=(1, ), dtype=flow.int32
            )

            return res_image, label

        if __name__ == "__main__":
            images, labels = ofrecord_reader_job()
            # image.shape (16, 224, 224, 3)

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
    images: oneflow_api.BlobDesc,
    target_size: int,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    resize_side: str = "shorter",
    interpolation_type: str = "auto",
    name: Optional[str] = None,
) -> Sequence[oneflow_api.BlobDesc]:
    """This operator resizes image to target size. 

    Args:
        images (oneflow_api.BlobDesc): The input Blob. Its type should be `kTensorBuffer`. More details please refer to the code example. 
        target_size (int): An int, the target size. 
        min_size (Optional[int], optional): If `min_size` is not None, the shorter side must be greater than or equal to `min_size`. Default is None. Defaults to None.
        max_size (Optional[int], optional): If `max_size` is not None, the longer side must be less than or equal to `max_size`. Defaults to None.
        resize_side (str, optional): A str of "longer" or "shorter". Only works when `keep_aspect_ratio` is True. If `resize_side` is "longer", the longer side of image will be resized to `target_size`. If `resize_side` is "shorter", the shorter side of image will be resized to `target_size`. Defaults to "shorter".
        interpolation_type (str, optional): A str of "auto", "bilinear", "nearest_neighbor", "bicubic" or "area". Indicate interpolation method used to resize image. Defaults to "auto".
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        Sequence[oneflow_api.BlobDesc]: A Sequence includes the result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple
        import numpy as np
        import cv2


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            assert image_static_shape[0] == 1, str(image_static_shape)
            image_static_shape[0] = len(image_shapes)
            return image_static_shape

        def _of_image_target_resize(images, image_static_shape, target_size, max_size):
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_target_resize_job(images_def: tp.ListListNumpy.Placeholder(shape=image_static_shape, dtype=flow.float)
            ) -> Tuple[tp.ListListNumpy, tp.ListNumpy, tp.ListNumpy]: 
                # The input Blob type should be "kTensorBuffer"
                # So we use oneflow.tensor_list_to_tensor_buffer to convert
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)

                resized_images_buffer, size, scale = flow.image_target_resize(
                    images_buffer,
                    target_size=target_size,
                    max_size=max_size,
                    resize_side="shorter",
                )
                # We convert back to "tensorlist" type
                resized_images = flow.tensor_buffer_to_tensor_list(
                    resized_images_buffer,
                    shape=(target_size, max_size, image_static_shape[-1]),
                    dtype=flow.float,
                )
                return resized_images, size, scale

            resized_images, size, scale = image_target_resize_job([images])
            resized_image = resized_images[0]
            size = size[0]
            scale = scale[0]

            return resized_images, size, scale

        if __name__ == "__main__": 
            img = _read_images_by_cv(['./img/1.jpg'])
            img_shape = _get_images_static_shape(img) # In example is [1, 349, 367, 3]
            target_size = 256
            max_size = 512
            resized_images, size, scale = _of_image_target_resize(img, tuple(img_shape), target_size, max_size)
            # Here the shorter side is "349", we resize it to target_size(256)
            # The scale is 256 / 349 = 0.73
            # The longer side will be resized to 367 * scale = 269
            # get the first element from the resized_images (its type is `list.list`)
            print(resized_images[0][0].shape) # (1, 256, 269, 3)

    """
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
    input_blob: oneflow_api.BlobDesc,
    mirror_blob: Optional[oneflow_api.BlobDesc] = None,
    color_space: str = "BGR",
    output_layout: str = "NCHW",
    crop_h: int = 0,
    crop_w: int = 0,
    crop_pos_y: float = 0.5,
    crop_pos_x: float = 0.5,
    mean: Sequence[float] = [0.0],
    std: Sequence[float] = [1.0],
    output_dtype: flow.dtype = flow.float,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator performs the cropping, normalization, and horizontal flip for input Blob. 

    If `crop_h` and `crop_w` are provided, the image cropping position is specified by "crop_pos_y" and "crop_pos_x". 

    The position is computed as follows: 

    .. math:: 

        & crop_x = crop\_pos\_x*(Width-crop\_w)

        & crop_y = crop\_pos\_y*(Height-crop\_h)

    The `Width` and `Height` is the width and height of input Blob. 

    Args:
        input_blob (oneflow_api.BlobDesc): The input Blob. 
        mirror_blob (Optional[oneflow_api.BlobDesc], optional): The operation for horizontal flip, if it is `None`, the operator will not perform the horizontal flip. Defaults to None.
        color_space (str, optional): The color space for input Blob. Defaults to "BGR".
        output_layout (str, optional): The output format. Defaults to "NCHW".
        crop_h (int, optional): The image cropping window height. Defaults to 0.
        crop_w (int, optional): The image cropping window width. Defaults to 0.
        crop_pos_y (float, optional): The vertical position of the image cropping window, the value range is normalized to (0.0, 1.0). Defaults to 0.5.
        crop_pos_x (float, optional): The horizontal position of the image cropping window, the value range is normalized to (0.0, 1.0). Defaults to 0.5.
        mean (Sequence[float], optional): The mean value for normalization. Defaults to [0.0].
        std (Sequence[float], optional): The standard deviation values for normalization. Defaults to [1.0].
        output_dtype (flow.dtype, optional): The datatype of output Blob. Defaults to flow.float.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Raises:
        NotImplementedError: The data type of input Blob should be `tensor_buffer` or `uint8`

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def crop_mirror_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 1
            color_space = "RGB"
            # our ofrecord file path is "./dataset/part-0"
            ofrecord = flow.data.ofrecord_reader(
                "./imgdataset",
                batch_size=batch_size,
                data_part_num=1,
                part_name_suffix_length=-1,
                part_name_prefix='part-', 
                shuffle_after_epoch=True,
            )
            image = flow.data.OFRecordImageDecoder(
                    ofrecord, "encoded", color_space=color_space
                )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(512, 512)
                )
            label = flow.data.OFRecordRawDecoder(
                ofrecord, "class/label", shape=(1, ), dtype=flow.int32
            )
            rng = flow.random.CoinFlip(batch_size=batch_size)
            normal = flow.image.CropMirrorNormalize(
                    res_image,
                    mirror_blob=rng,
                    color_space=color_space,
                    crop_h= 256,
                    crop_w= 256,
                    crop_pos_y=0.5,
                    crop_pos_x=0.5,
                    mean=[123.68, 116.779, 103.939],
                    std=[58.393, 57.12, 57.375],
                    output_dtype=flow.float,
                )

            return normal, label

        if __name__ == "__main__":
            images, labels = crop_mirror_job()
            # images.shape (1, 3, 256, 256)

    """
    if name is None:
        name = id_util.UniqueStr("CropMirrorNormalize_")
    op_type_name = ""
    if input_blob.dtype is flow.tensor_buffer:
        op_type_name = "crop_mirror_normalize_from_tensorbuffer"
    elif input_blob.dtype is flow.uint8:
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
    input_blob: oneflow_api.BlobDesc,
    num_attempts: int = 10,
    seed: Optional[int] = None,
    random_area: Sequence[float] = None,
    random_aspect_ratio: Sequence[float] = None,
    name: str = "ImageRandomCrop",
) -> oneflow_api.BlobDesc:
    """This operator crops the input image randomly. 

    Args:
        input_blob (oneflow_api.BlobDesc): The input Blob. 
        num_attempts (int, optional): The maximum number of random cropping attempts. Defaults to 10.
        seed (Optional[int], optional): The random seed. Defaults to None.
        random_area (Sequence[float], optional): The random cropping area. Defaults to None.
        random_aspect_ratio (Sequence[float], optional): The random scaled ratio. Defaults to None.
        name (str, optional): The name for the operation. Defaults to "ImageRandomCrop".

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np
        import cv2


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            assert image_static_shape[0] == 1, str(image_static_shape)
            image_static_shape[0] = len(image_shapes)
            return image_static_shape

        def _of_image_random_crop(images, image_static_shape):
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_random_crop_job(images_def: tp.ListListNumpy.Placeholder(shape=image_static_shape, dtype=flow.float)
            ) -> tp.ListListNumpy: 
                # The input Blob type should be "kTensorBuffer"
                # So we use oneflow.tensor_list_to_tensor_buffer to convert
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
                # Do the random crop
                random_crop_buffer = flow.image.random_crop(
                    images_buffer,
                    random_area=[0.15, 0.80],
                    random_aspect_ratio=[0.75, 1.55],
                )
                # We convert back to "tensorlist" type
                random_crop_images = flow.tensor_buffer_to_tensor_list(
                    random_crop_buffer,
                    shape=(image_static_shape[1], image_static_shape[2], image_static_shape[-1]),
                    dtype=flow.float,
                )
                return random_crop_images

            random_crop_images = image_random_crop_job([images])

            return random_crop_images

        if __name__ == "__main__": 
            img = _read_images_by_cv(['./img/1.jpg'])
            img_shape = _get_images_static_shape(img) # In example is (1, 234, 346, 3)
            random_crop_images = _of_image_random_crop(img, tuple(img_shape)) 
            # random_crop_images.shape is (234, 346, 3)

    """
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

    def forward(self, input: oneflow_api.BlobDesc):
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
) -> oneflow_api.BlobDesc:
    """This operator performs the horizontal flip. 

    Args:
        batch_size (int, optional): The batch size. Defaults to 1.
        seed (Optional[int], optional): The random seed. Defaults to None.
        probability (float, optional): The flip probability. Defaults to 0.5.
        name (str, optional): The name for the operation. Defaults to "CoinFlip".

    Returns:
        oneflow_api.BlobDesc: [description]

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def coin_flip_job() -> Tuple[tp.Numpy, tp.Numpy]:
            batch_size = 1
            color_space = "RGB"
            # our ofrecord file path is "./dataset/part-0"
            ofrecord = flow.data.ofrecord_reader(
                "./imgdataset",
                batch_size=batch_size,
                data_part_num=1,
                part_name_suffix_length=-1,
                part_name_prefix='part-', 
                shuffle_after_epoch=True,
            )
            image = flow.data.OFRecordImageDecoder(
                    ofrecord, "encoded", color_space=color_space
                )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(512, 512)
                )
            label = flow.data.OFRecordRawDecoder(
                ofrecord, "class/label", shape=(1, ), dtype=flow.int32
            )
            coin_flip = flow.random.CoinFlip(
                batch_size=batch_size, 
                probability=0.8
            )
            normal = flow.image.CropMirrorNormalize(
                    res_image,
                    mirror_blob=coin_flip,
                    color_space=color_space,
                    crop_h= 256,
                    crop_w= 256,
                    crop_pos_y=0.5,
                    crop_pos_x=0.5,
                    mean=[123.68, 116.779, 103.939],
                    std=[58.393, 57.12, 57.375],
                    output_dtype=flow.float,
                )

            return normal, label

        if __name__ == "__main__":
            images, labels = coin_flip_job()

    """
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
    images_bytes_buffer: oneflow_api.BlobDesc,
    dtype: flow.dtype = flow.uint8,
    color_space: str = "BGR",
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator decode the image. 

    Args:
        images_bytes_buffer (oneflow_api.BlobDesc): The input Blob. Its type should be `kTensorBuffer`. More details please refer to the code example. 
        dtype (flow.dtype, optional): The data type. Defaults to flow.uint8.
        color_space (str, optional): The color space. Defaults to "BGR".
        name (Optional[str], optional): The name for the opreation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The decoded image list. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        import numpy as np
        from PIL import Image


        def _of_image_decode(images):
            image_files = [open(im, "rb") for im in images]
            images_bytes = [imf.read() for imf in image_files]
            static_shape = (len(images_bytes), max([len(bys) for bys in images_bytes]))
            for imf in image_files:
                imf.close()

            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_decode_job(
                images_def: tp.ListListNumpy.Placeholder(shape=static_shape, dtype=flow.int8)
            )->tp.ListListNumpy:
                # convert to tensor buffer
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
                decoded_images_buffer = flow.image_decode(images_buffer)
                # Remember to set a shape
                # convert back to tensor list
                return flow.tensor_buffer_to_tensor_list(
                    decoded_images_buffer, shape=(640, 640, 3), dtype=flow.uint8
                )

            images_np_arr = [
                np.frombuffer(bys, dtype=np.byte).reshape(1, -1) for bys in images_bytes
            ]
            decoded_images = image_decode_job([images_np_arr])
            return decoded_images[0]


        if __name__ == "__main__": 
            img = _of_image_decode(['./img/1.jpg'])
            print(img[0].shape) # Our image shape is (1, 349, 367, 3)

    """
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
    images: oneflow_api.BlobDesc,
    shape: Sequence[int],
    dtype: flow.dtype,
    alignment: int,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    r"""This operator aligns the shape for a batch of images. 

    The aligned shape is computed as: 

    .. math:: 

        & shape_{width} = int(\frac{(shape_{width}+alignment-1)}{alignment})*alignment

        & shape_{height} = int(\frac{(shape_{height}+alignment-1)}{alignment})*alignment

    Args:
        images (oneflow_api.BlobDesc): The images. 
        shape (Sequence[int]): The maximum static shape of input images. 
        dtype (flow.dtype): The data type. 
        alignment (int): The align factor. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import cv2
        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp 


        def _of_image_batch_align(images, input_shape, output_shape, alignment):
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_batch_align_job(
                images_def: tp.ListListNumpy.Placeholder(shape=input_shape, dtype=flow.float)
            ) -> tp.ListNumpy:
                # Convert to tensor buffer
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
                image = flow.image_batch_align(
                    images_buffer, shape=output_shape[1:], dtype=flow.float, alignment=alignment
                )
                return image

            image = image_batch_align_job([images])
            return image[0]


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            assert image_static_shape[0] == 1, str(image_static_shape)
            image_static_shape[0] = len(image_shapes)
            return image_static_shape

        def _roundup(x, n):
            # compute the aligned shape
            return int((x + n - 1) / n) * n

        if __name__ == "__main__": 
            img = _read_images_by_cv(['./img/1.jpg', './img/2.jpg', './img/3.jpg'])
            img_shape = _get_images_static_shape(img) # In example is [3, 349, 367, 3]
            alignment = 16 # alignment factor
            aligned_image_shape = [
                img_shape[0],
                _roundup(img_shape[1], alignment),
                _roundup(img_shape[2], alignment),
                img_shape[3],
            ]
            image = _of_image_batch_align(img, tuple(img_shape), aligned_image_shape, alignment)

    """
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
    image: oneflow_api.BlobDesc,
    std: Sequence[float],
    mean: Sequence[float],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator normalizes the image. 

    Args:
        image (oneflow_api.BlobDesc): The input image. 
        std (Sequence[float]): The standard deviation of the images. 
        mean (Sequence[float]): The mean value of the images. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import cv2
        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp 


        def _of_image_normalize(images, image_shape, std, mean):
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_normalize_job(
                images_def: tp.ListListNumpy.Placeholder(shape=image_shape, dtype=flow.float)
            ) -> tp.ListListNumpy:
                # Convert to tensor buffer
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
                # Normalize the imagess
                norm_images = flow.image_normalize(images_buffer, std, mean)
                # Convert back to tensor list
                return flow.tensor_buffer_to_tensor_list(
                    norm_images, shape=image_shape[1:], dtype=flow.float
                )

            image_tensor = image_normalize_job([images])
            return image_tensor[0]


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            assert image_static_shape[0] == 1, str(image_static_shape)
            image_static_shape[0] = len(image_shapes)
            return image_static_shape

        if __name__ == "__main__": 
            img = _read_images_by_cv(['./img/1.jpg', './img/2.jpg', './img/3.jpg'])
            img_shape = _get_images_static_shape(img) # In example is [3, 349, 367, 3]
            image = _of_image_normalize(img, 
                                        tuple(img_shape), 
                                        std=(102.9801, 115.9465, 122.7717),
                                        mean=(1.0, 1.0, 1.0))

    """
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
    image: oneflow_api.BlobDesc,
    flip_code: Union[int, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator flips the images. 

    The flip code corresponds to the different flip mode: 

    0 (0x00): Non Flip 

    1 (0x01): Horizontal Flip 

    16 (0x10): Vertical Flip

    17 (0x11): Both Horizontal and Vertical Flip

    Args:
        image (oneflow_api.BlobDesc): The input images. 
        flip_code (Union[int, oneflow_api.BlobDesc]): The flip code. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob

    For example: 

    .. code-block:: python 

        import cv2
        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp 


        def _of_image_flip(images, image_shape, flip_code):
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def image_flip_job(
                images_def: tp.ListListNumpy.Placeholder(shape=image_shape, dtype=flow.float)
            ) -> tp.ListListNumpy:
                images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
                flip_images = flow.image_flip(images_buffer, flip_code)
                return flow.tensor_buffer_to_tensor_list(
                    flip_images, shape=image_shape[1:], dtype=flow.float
                )

            image_tensor = image_flip_job([images])
            return image_tensor[0]


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            assert image_static_shape[0] == 1, str(image_static_shape)
            image_static_shape[0] = len(image_shapes)
            return image_static_shape

        if __name__ == "__main__": 
            img = _read_images_by_cv(['./img/1.jpg', './img/2.jpg', './img/3.jpg'])
            img_shape = _get_images_static_shape(img) # In example is [3, 349, 367, 3]
            image = _of_image_flip(img, 
                           tuple(img_shape), 
                           flip_code=1)

    """
    assert isinstance(image, oneflow_api.BlobDesc)

    if name is None:
        name = id_util.UniqueStr("ImageFlip_")

    if not isinstance(flip_code, oneflow_api.BlobDesc):
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
    bbox: oneflow_api.BlobDesc,
    image_size: oneflow_api.BlobDesc,
    flip_code: Union[int, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator flips the object bounding box. 

    The flip code corresponds to the different flip mode: 

    0 (0x00): Non Flip 

    1 (0x01): Horizontal Flip 

    16 (0x10): Vertical Flip

    17 (0x11): Both Horizontal and Vertical Flip

    Args:
        bbox (oneflow_api.BlobDesc): The bounding box. 
        image_size (oneflow_api.BlobDesc): The size of input image. 
        flip_code (Union[int, oneflow_api.BlobDesc]): The flip code. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob 

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp 


        def _of_object_bbox_flip(bbox_list, image_size, flip_code):
            bbox_shape = _get_bbox_static_shape(bbox_list)
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def object_bbox_flip_job(
                bbox_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(bbox_shape), dtype=flow.float
                ),
                image_size_def: tp.ListNumpy.Placeholder(
                    shape=image_size.shape, dtype=flow.int32
                ),
            ) -> tp.ListListNumpy:
                bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
                flip_bbox = flow.object_bbox_flip(bbox_buffer, image_size_def, flip_code)
                return flow.tensor_buffer_to_tensor_list(
                    flip_bbox, shape=bbox_shape[1:], dtype=flow.float
                )

            input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
            bbox_tensor = object_bbox_flip_job([input_bbox_list], [image_size])
            return bbox_tensor[0]


        def _get_bbox_static_shape(bbox_list):
            bbox_shapes = [bbox.shape for bbox in bbox_list]
            bbox_static_shape = np.amax(bbox_shapes, axis=0)
            assert isinstance(
                bbox_static_shape, np.ndarray
            ), "bbox_shapes: {}, bbox_static_shape: {}".format(
                str(bbox_shapes), str(bbox_static_shape)
            )
            bbox_static_shape = bbox_static_shape.tolist()
            bbox_static_shape.insert(0, len(bbox_list))
            return bbox_static_shape

        if __name__ == "__main__": 
            bbox = np.array([[[20.0, 40.0, 80.0, 160.0],  
                            [30.0, 50.0, 70.0, 100.0]]]).astype(np.single) # [x1, y1, x2, y2]
            image_size = np.array([[480, 620]]).astype(np.int32)
            bbox_flip =  _of_object_bbox_flip(bbox, 
                                            image_size, 
                                            flip_code=1) # Horizontal Flip
            print(bbox_flip[0][0])

            # [[399.  40. 459. 160.]
            #  [409.  50. 449. 100.]]
    """
    assert isinstance(bbox, oneflow_api.BlobDesc)
    assert isinstance(image_size, oneflow_api.BlobDesc)
    assert bbox.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectBboxFlip_")

    if not isinstance(flip_code, oneflow_api.BlobDesc):
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
    bbox: oneflow_api.BlobDesc, scale: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator scales the input image and the corresponding bounding box. It returns the scaled bounding box. 

    Args:
        bbox (oneflow_api.BlobDesc): The bounding box. 
        scale (oneflow_api.BlobDesc): The scale factor. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob.

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp
        import cv2 
        from typing import Tuple 


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return images 


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            image_static_shape.insert(0, len(image_shapes))
            return image_static_shape


        def _get_bbox_static_shape(bbox_list):
            bbox_shapes = [bbox.shape for bbox in bbox_list]
            bbox_static_shape = np.amax(bbox_shapes, axis=0)
            assert isinstance(
                bbox_static_shape, np.ndarray
            ), "bbox_shapes: {}, bbox_static_shape: {}".format(
                str(bbox_shapes), str(bbox_static_shape)
            )
            bbox_static_shape = bbox_static_shape.tolist()
            bbox_static_shape.insert(0, len(bbox_list))
            return bbox_static_shape


        def _of_target_resize_bbox_scale(images, bbox_list, target_size, max_size):
            image_shape = _get_images_static_shape(images)
            bbox_shape = _get_bbox_static_shape(bbox_list)

            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def target_resize_bbox_scale_job(
                image_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(image_shape), dtype=flow.float
                ),
                bbox_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(bbox_shape), dtype=flow.float
                ),
            ) -> Tuple[tp.ListListNumpy, tp.ListNumpy]:
                images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
                resized_images_buffer, new_size, scale = flow.image_target_resize(
                    images_buffer, target_size=target_size, max_size=max_size
                )
                bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
                scaled_bbox = flow.object_bbox_scale(bbox_buffer, scale)
                scaled_bbox_list = flow.tensor_buffer_to_tensor_list(
                    scaled_bbox, shape=bbox_shape[1:], dtype=flow.float
                )
                return scaled_bbox_list, new_size

            input_image_list = [np.expand_dims(image, axis=0) for image in images]
            input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
            output_bbox_list, output_image_size = target_resize_bbox_scale_job(
                [input_image_list], [input_bbox_list]
            )
            return output_bbox_list[0], output_image_size[0]


        if __name__ == "__main__": 
            images = _read_images_by_cv(['./img/1.jpg', './img/2.jpg'])
            bbox = np.array([[[20.0, 40.0, 80.0, 160.0],  
                            [30.0, 50.0, 70.0, 100.0]], 
                            [[26.0, 40.0, 86.0, 160.0],  
                            [36.0, 56.0, 76.0, 106.0]]]).astype(np.single) # [x1, y1, x2, y2]
            bbox, size = _of_target_resize_bbox_scale(images, bbox, 280, 350)
            print(bbox[0])
            print(bbox[1])

            # [[[ 16.0218    32.09169   64.0872   128.36676 ]
            #   [ 24.032698  40.114613  56.076298  80.229225]]]

            # [[[ 24.186047  37.170418  80.       148.68167 ]
            #   [ 33.488373  52.038586  70.69768   98.5016  ]]]

    """
    assert isinstance(bbox, oneflow_api.BlobDesc)
    assert isinstance(scale, oneflow_api.BlobDesc)
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
    poly: oneflow_api.BlobDesc,
    image_size: oneflow_api.BlobDesc,
    flip_code: Union[int, oneflow_api.BlobDesc],
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator flips the segmentation points in image. 

    The flip code corresponds to the different flip mode: 

    0 (0x00): Non Flip 

    1 (0x01): Horizontal Flip 

    16 (0x10): Vertical Flip

    17 (0x11): Both Horizontal and Vertical Flip

    Args:
        poly (oneflow_api.BlobDesc): The poly segmentation points. 
        image_size (oneflow_api.BlobDesc): The image size. 
        flip_code (Union[int, oneflow_api.BlobDesc]): The filp code. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob 

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp
        import cv2 


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return [np.expand_dims(image, axis=0) for image in images]


        def _of_object_segm_poly_flip(poly_list, image_size, flip_code):
            poly_shape = _get_segm_poly_static_shape(poly_list)

            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def object_segm_poly_flip_job(
                poly_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(poly_shape), dtype=flow.float
                ),
                image_size_def: tp.ListNumpy.Placeholder(
                    shape=image_size.shape, dtype=flow.int32
                ),
            ) -> tp.ListListNumpy:
                poly_buffer = flow.tensor_list_to_tensor_buffer(poly_def)
                flip_poly = flow.object_segmentation_polygon_flip(
                    poly_buffer, image_size_def, flip_code
                )
                return flow.tensor_buffer_to_tensor_list(
                    flip_poly, shape=poly_shape[1:], dtype=flow.float
                )

            input_poly_list = [np.expand_dims(poly, axis=0) for poly in poly_list]
            poly_tensor = object_segm_poly_flip_job([input_poly_list], [image_size])
            return poly_tensor[0]


        def _get_segm_poly_static_shape(poly_list):
            poly_shapes = [poly.shape for poly in poly_list]
            poly_static_shape = np.amax(poly_shapes, axis=0)
            assert isinstance(
                poly_static_shape, np.ndarray
            ), "poly_shapes: {}, poly_static_shape: {}".format(
                str(poly_shapes), str(poly_static_shape)
            )
            poly_static_shape = poly_static_shape.tolist()
            poly_static_shape.insert(0, len(poly_list))
            return poly_static_shape

        if __name__ == "__main__": 
            segm_poly_list = []
            segmentations = [[[20.0, 40.0], [80.0, 160.0], [100.0, 210.0]], # Image 1 segmentation point
                            [[25.0, 45.0], [85.0, 165.0], [105.0, 215.0]]] # Image 2 segmentation point
            for segmentation in segmentations: 
                polygon = []
                for seg in segmentation: 
                    polygon.extend(seg)
                poly_array = np.array(polygon, dtype=np.single).reshape(-1, 2) # Reshape it
                segm_poly_list.append(poly_array)

            image_size = np.array([[480, 620], # Image 1 size
                                [640, 640]]).astype(np.int32) # Image 2 size
            of_segm_poly_list = _of_object_segm_poly_flip(
                segm_poly_list, image_size, flip_code=1
            ) # Horizontal Flip
            print(of_segm_poly_list[0])
            print(of_segm_poly_list[1])

            # of_segm_poly_list[0]
            # [[[460.  40.]
            #   [400. 160.]
            #   [380. 210.]]]

            # of_segm_poly_list[1]
            # [[[615.  45.]
            #   [555. 165.]
            #   [535. 215.]]]

    """
    assert isinstance(poly, oneflow_api.BlobDesc)
    assert isinstance(image_size, oneflow_api.BlobDesc)
    assert poly.shape[0] == image_size.shape[0]

    if name is None:
        name = id_util.UniqueStr("ObjectSegmPolyFilp_")

    if not isinstance(flip_code, oneflow_api.BlobDesc):
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
    poly: oneflow_api.BlobDesc, scale: oneflow_api.BlobDesc, name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator scales the segmentation points in the images. 

    Args:
        poly (oneflow_api.BlobDesc): The poly segmentation points. 
        scale (oneflow_api.BlobDesc): The image scale. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp
        import cv2 
        from typing import Tuple 


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return images


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            image_static_shape.insert(0, len(image_shapes))
            return image_static_shape


        def _get_segm_poly_static_shape(poly_list):
            poly_shapes = [poly.shape for poly in poly_list]
            poly_static_shape = np.amax(poly_shapes, axis=0)
            assert isinstance(
                poly_static_shape, np.ndarray
            ), "poly_shapes: {}, poly_static_shape: {}".format(
                str(poly_shapes), str(poly_static_shape)
            )
            poly_static_shape = poly_static_shape.tolist()
            poly_static_shape.insert(0, len(poly_list))
            return poly_static_shape


        def _get_bbox_static_shape(bbox_list):
            bbox_shapes = [bbox.shape for bbox in bbox_list]
            bbox_static_shape = np.amax(bbox_shapes, axis=0)
            assert isinstance(
                bbox_static_shape, np.ndarray
            ), "bbox_shapes: {}, bbox_static_shape: {}".format(
                str(bbox_shapes), str(bbox_static_shape)
            )
            bbox_static_shape = bbox_static_shape.tolist()
            bbox_static_shape.insert(0, len(bbox_list))
            return bbox_static_shape


        def _of_object_segm_poly_scale(images, poly_list, target_size, max_size):
            image_shape = _get_images_static_shape(images)
            print(image_shape)
            poly_shape = _get_segm_poly_static_shape(poly_list)
            print("Poly shape is ", poly_shape)
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.default_logical_view(flow.scope.mirrored_view())

            @flow.global_function(function_config=func_config)
            def object_segm_poly_scale_job(
                image_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(image_shape), dtype=flow.float
                ),
                poly_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(poly_shape), dtype=flow.float
                ),
            ) -> Tuple[tp.ListListNumpy, tp.ListNumpy]:
                images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
                resized_images_buffer, new_size, scale = flow.image_target_resize(
                    images_buffer, target_size=target_size, max_size=max_size
                )
                poly_buffer = flow.tensor_list_to_tensor_buffer(poly_def)
                scaled_poly = flow.object_segmentation_polygon_scale(poly_buffer, scale)
                scaled_poly_list = flow.tensor_buffer_to_tensor_list(
                    scaled_poly, shape=poly_shape[1:], dtype=flow.float
                )
                return scaled_poly_list, new_size

            input_image_list = [np.expand_dims(image, axis=0) for image in images]
            input_poly_list = [np.expand_dims(poly, axis=0) for poly in poly_list]

            output_poly_list, output_image_size = object_segm_poly_scale_job(
                [input_image_list], [input_poly_list]
            )

            return output_poly_list[0], output_image_size

        if __name__ == "__main__": 
            images = _read_images_by_cv(['./img/1.jpg', './img/2.jpg'])
            segm_poly_list = []
            segmentations = [[[20.0, 40.0], [80.0, 160.0], [100.0, 210.0]], # Image 1 segmentation point
                            [[25.0, 45.0], [85.0, 165.0], [105.0, 215.0]]] # Image 2 segmentation point

            for segmentation in segmentations: 
                polygon = []
                for seg in segmentation: 
                    polygon.extend(seg)
                poly_array = np.array(polygon, dtype=np.single).reshape(-1, 2) # Reshape it
                segm_poly_list.append(poly_array)

            bbox, size = _of_object_segm_poly_scale(images, segm_poly_list, 280, 350)

    """
    assert isinstance(poly, oneflow_api.BlobDesc)
    assert isinstance(scale, oneflow_api.BlobDesc)
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
    poly: oneflow_api.BlobDesc,
    poly_index: oneflow_api.BlobDesc,
    image_size: oneflow_api.BlobDesc,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    """This operator converts the poly segment points to the segment mask array. 

    Args:
        poly (oneflow_api.BlobDesc): The poly segment points. 
        poly_index (oneflow_api.BlobDesc): The poly segment index. 
        image_size (oneflow_api.BlobDesc): The input image size. 
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    .. code-block:: python 

        import numpy as np
        import oneflow as flow
        import oneflow.typing as tp
        import cv2 
        from typing import Tuple 


        def _read_images_by_cv(image_files):
            images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
            return images


        def _get_images_static_shape(images):
            image_shapes = [image.shape for image in images]
            image_static_shape = np.amax(image_shapes, axis=0)
            assert isinstance(
                image_static_shape, np.ndarray
            ), "image_shapes: {}, image_static_shape: {}".format(
                str(image_shapes), str(image_static_shape)
            )
            image_static_shape = image_static_shape.tolist()
            image_static_shape.insert(0, len(image_shapes))
            return image_static_shape


        def _get_segm_poly_static_shape(poly_list, poly_index_list):
            assert len(poly_list) == len(poly_index_list)
            num_images = len(poly_list)
            max_poly_elems = 0
            for poly, poly_index in zip(poly_list, poly_index_list):
                assert len(poly.shape) == 2
                assert len(poly_index.shape) == 2, str(poly_index.shape)
                assert poly.shape[0] == poly_index.shape[0]
                assert poly.shape[1] == 2
                assert poly_index.shape[1] == 3
                max_poly_elems = max(max_poly_elems, poly.shape[0])
            return [num_images, max_poly_elems, 2], [num_images, max_poly_elems, 3]

        def _segm_poly_to_tensor(img_segm_poly_list):
            poly_array_list = []
            poly_index_array_list = []
            for img_idx, segm_poly_list in enumerate(img_segm_poly_list):
                img_poly_elem_list = []
                img_poly_index_list = []

                for obj_idx, poly_list in enumerate(segm_poly_list):
                    for poly_idx, poly in enumerate(poly_list):
                        img_poly_elem_list.extend(poly)
                        for pt_idx, pt in enumerate(poly):
                            if pt_idx % 2 == 0:
                                img_poly_index_list.append([pt_idx / 2, poly_idx, obj_idx])

                img_poly_array = np.array(img_poly_elem_list, dtype=np.single).reshape(-1, 2)
                assert img_poly_array.size > 0, segm_poly_list
                poly_array_list.append(img_poly_array)

                img_poly_index_array = np.array(img_poly_index_list, dtype=np.int32)
                assert img_poly_index_array.size > 0, segm_poly_list
                poly_index_array_list.append(img_poly_index_array)

            return poly_array_list, poly_index_array_list


        def _of_poly_to_mask_pipline(
            images, poly_list, poly_index_list, num_segms_list, target_size, max_size
        ):  
            print(len(images))
            print(len(poly_list))

            assert len(images) == len(poly_list)
            assert len(poly_list) == len(poly_index_list)
            image_shape = _get_images_static_shape(images)
            poly_shape, poly_index_shape = _get_segm_poly_static_shape(
                poly_list, poly_index_list
            )
            max_num_segms = max(num_segms_list)

            func_config = flow.FunctionConfig()
            func_config.default_logical_view(flow.scope.mirrored_view())
            func_config.default_data_type(flow.float)


            @flow.global_function(function_config=func_config)
            def poly_to_mask_job(
                image_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(image_shape), dtype=flow.float
                ),
                poly_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(poly_shape), dtype=flow.float
                ),
                poly_index_def: tp.ListListNumpy.Placeholder(
                    shape=tuple(poly_index_shape), dtype=flow.int32
                ),
            ) -> Tuple[tp.ListListNumpy, tp.ListListNumpy]:
                images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
                resized_images_buffer, new_size, scale = flow.image_target_resize(
                    images_buffer, target_size=target_size, max_size=max_size
                )
                poly_buffer = flow.tensor_list_to_tensor_buffer(poly_def)
                poly_index_buffer = flow.tensor_list_to_tensor_buffer(poly_index_def)
                scaled_poly_buffer = flow.object_segmentation_polygon_scale(poly_buffer, scale)
                mask_buffer = flow.object_segmentation_polygon_to_mask(
                    scaled_poly_buffer, poly_index_buffer, new_size
                )
                mask_list = flow.tensor_buffer_to_tensor_list(
                    mask_buffer, shape=(max_num_segms, target_size, max_size), dtype=flow.int8
                )
                scaled_poly_list = flow.tensor_buffer_to_tensor_list(
                    scaled_poly_buffer, shape=poly_shape[1:], dtype=flow.float
                )
                return mask_list, scaled_poly_list

            input_image_list = [np.expand_dims(image, axis=0) for image in images]
            input_poly_list = [np.expand_dims(poly, axis=0) for poly in poly_list]
            input_poly_index_list = [
                np.expand_dims(poly_index, axis=0) for poly_index in poly_index_list
            ]

            output_mask_list, output_poly_list = poly_to_mask_job(
                [input_image_list], [input_poly_list], [input_poly_index_list]
            )

            return output_mask_list[0], output_poly_list[0]

        if __name__ == "__main__": 
            images = _read_images_by_cv(['./img/1.jpg', './img/2.jpg'])
            segm_poly_list = []

            segmentations = [[[20.0, 40.0, 80.0, 160.0, 100.0, 210.0, 120.0, 215.0]], # Image 1 segmentation point
                            [[24.0, 42.0, 86.0, 168.0, 103.0, 223.0, 125.0, 235.0]]] # Image 2 segmentation point

            for segmentation in segmentations: 
                polygon = []
                for seg in segmentation: 
                    polygon.extend(seg)

                poly_array = np.array(polygon, dtype=np.single).reshape(-1, 2) # Reshape it
                segm_poly_list.append([poly_array])

            poly_list, poly_index_list = _segm_poly_to_tensor(segm_poly_list)
            num_segms_list = [len(segm_poly_list) for segm_poly_list in segm_poly_list]
            target_size = 280
            max_size = 350
            of_mask_list, of_scaled_poly_list = _of_poly_to_mask_pipline(
                images, poly_list, poly_index_list, num_segms_list, target_size, max_size
            )
            of_mask_list = [
                mask_array.reshape(-1, mask_array.shape[-2], mask_array.shape[-1])
                for mask_array in of_mask_list
            ] # reshape it 

    """
    assert isinstance(poly, oneflow_api.BlobDesc)
    assert isinstance(poly_index, oneflow_api.BlobDesc)
    assert isinstance(image_size, oneflow_api.BlobDesc)
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
    remove_images_without_annotations: bool = True,
    name: str = None,
) -> oneflow_api.BlobDesc:
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
            remove_images_without_annotations=remove_images_without_annotations,
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
        remove_images_without_annotations: bool = True,
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
            .Attr("session_id", flow.current_scope().session_id)
            .Attr("annotation_file", annotation_file)
            .Attr("image_dir", image_dir)
            .Attr("batch_size", batch_size)
            .Attr("shuffle_after_epoch", shuffle)
            .Attr("random_seed", random_seed)
            .Attr("group_by_ratio", group_by_aspect_ratio)
            .Attr(
                "remove_images_without_annotations", remove_images_without_annotations
            )
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
) -> oneflow_api.BlobDesc:
    """This operator creates a reader for image classification tasks. 

    Args:
        ofrecord_dir (str): The directory of ofrecord file. 
        image_feature_name (str): The name of the image feature. 
        label_feature_name (str): The name of the label feature. 
        batch_size (int, optional): The batch_size. Defaults to 1.
        data_part_num (int, optional): The amounts of data part. Defaults to 1.
        part_name_prefix (str, optional): The prefix of data part name. Defaults to "part-".
        part_name_suffix_length (int, optional): The suffix name of data part name. Defaults to -1.
        random_shuffle (bool, optional): Whether to random shuffle the data. Defaults to False.
        shuffle_buffer_size (int, optional): The buffer size for shuffle data. Defaults to 1024.
        shuffle_after_epoch (bool, optional): Whether to shuffle the data after each epoch. Defaults to False.
        color_space (str, optional): The color space. Defaults to "BGR".
        decode_buffer_size_per_thread (int, optional): The decode buffer size for per thread. Defaults to 32.
        num_decode_threads_per_machine (Optional[int], optional): The amounts of decode threads for each machine. Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp
        from typing import Tuple


        @flow.global_function(type="predict")
        def image_classifier_job() -> Tuple[tp.Numpy, tp.Numpy]:
            image, label = flow.data.ofrecord_image_classification_reader(
                ofrecord_dir="./imgdataset", 
                image_feature_name="encoded",
                label_feature_name="class/label",
                batch_size=8,
                data_part_num=1,
                part_name_prefix="part-",
                part_name_suffix_length=-1,
                random_shuffle=False,
                shuffle_after_epoch=False,
                color_space="RGB",
                decode_buffer_size_per_thread=16,
            )
            res_image, scale, new_size = flow.image.Resize(
                    image, target_size=(224, 224)
                )
            return res_image, label


        if __name__ == "__main__":
            images, labels = image_classifier_job()
            # images.shape (8, 224, 224, 3)

    """
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


@oneflow_export("data.OneRecDecoder", "data.onerec_decoder")
def OneRecDecoder(
    input_blob,
    key,
    dtype,
    shape,
    is_dynamic=False,
    reshape=None,
    batch_padding=None,
    name=None,
):
    if name is None:
        name = id_util.UniqueStr("OneRecDecoder_")
    if reshape is not None:
        has_reshape = True
    else:
        has_reshape = False
        reshape = shape
    if batch_padding is not None:
        has_batch_padding = True
    else:
        has_batch_padding = False
        batch_padding = shape
    return (
        flow.user_op_builder(name)
        .Op("onerec_decoder")
        .Input("in", [input_blob])
        .Output("out")
        .Attr("key", key)
        .Attr("data_type", dtype)
        .Attr("static_shape", shape)
        .Attr("is_dynamic", is_dynamic)
        .Attr("has_reshape", has_reshape)
        .Attr("reshape", reshape)
        .Attr("has_batch_padding", has_batch_padding)
        .Attr("batch_padding", batch_padding)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )
