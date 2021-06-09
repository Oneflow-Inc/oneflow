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
import oneflow as flow

from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from typing import Optional, List, Tuple, Sequence, Union
import traceback


def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True

    return seed, has_seed


@oneflow_export("nn.OfrecordReader")
@experimental_api
class OfrecordReader(Module):
    def __init__(
        self,
        ofrecord_dir: str,
        batch_size: int = 1,
        data_part_num: int = 1,
        part_name_prefix: str = "part-",
        part_name_suffix_length: int = -1,
        random_shuffle: bool = False,
        shuffle_buffer_size: int = 1024,
        shuffle_after_epoch: bool = False,
        random_seed: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__()
        seed, has_seed = mirrored_gen_random_seed(random_seed)
        self._op = (
            flow.builtin_op("OFRecordReader", name)
            .Output("out")
            .Attr("data_dir", ofrecord_dir)
            .Attr("data_part_num", data_part_num)
            .Attr("batch_size", batch_size)
            .Attr("part_name_prefix", part_name_prefix)
            .Attr("random_shuffle", random_shuffle)
            .Attr("shuffle_buffer_size", shuffle_buffer_size)
            .Attr("shuffle_after_epoch", shuffle_after_epoch)
            .Attr("part_name_suffix_length", part_name_suffix_length)
            .Attr("seed", seed)
            .Build()
        )

    def forward(self):
        res = self._op()[0]
        return res


@oneflow_export("nn.OfrecordRawDecoder")
@experimental_api
class OfrecordRawDecoder(Module):
    def __init__(
        self,
        blob_name: str,
        shape: Sequence[int],
        dtype: flow.dtype,
        dim1_varying_length: bool = False,
        auto_zero_padding: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        self._op = (
            flow.builtin_op("ofrecord_raw_decoder", name)
            .Input("in")
            .Output("out")
            .Attr("name", blob_name)
            .Attr("shape", shape)
            .Attr("data_type", dtype)
            .Attr("dim1_varying_length", dim1_varying_length)
            .Attr("auto_zero_padding", auto_zero_padding)
            .Build()
        )

    def forward(self, input):
        res = self._op(input)[0]
        return res


@oneflow_export("nn.CoinFlip")
@experimental_api
class CoinFlip(Module):
    def __init__(
        self,
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        probability: float = 0.5,
    ):
        super().__init__()
        seed, has_seed = mirrored_gen_random_seed(random_seed)
        self._op = (
            flow.builtin_op("coin_flip")
            .Output("out")
            .Attr("batch_size", batch_size)
            .Attr("probability", probability)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .Build()
        )

    def forward(self):
        res = self._op()[0]
        return res


@oneflow_export("nn.CropMirrorNormalize")
@experimental_api
class CropMirrorNormalize(Module):
    def __init__(
        self,
        color_space: str = "BGR",
        output_layout: str = "NCHW",
        crop_h: int = 0,
        crop_w: int = 0,
        crop_pos_y: float = 0.5,
        crop_pos_x: float = 0.5,
        mean: Sequence[float] = [0.0],
        std: Sequence[float] = [1.0],
        output_dtype: flow.dtype = flow.float,
    ):
        super().__init__()
        self._op = (
            flow.builtin_op("crop_mirror_normalize_from_uint8")
            .Input("in")
            .Input("mirror")
            .Output("out")
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
        )

        self._val_op = (
            flow.builtin_op("crop_mirror_normalize_from_tensorbuffer")
            .Input("in")
            .Output("out")
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
        )

    def forward(self, input, mirror=None):
        if mirror != None:
            res = self._op(input, mirror)[0]
        else:
            res = self._val_op(input)[0]
        return res


@oneflow_export("nn.OFRecordImageDecoderRandomCrop")
@experimental_api
class OFRecordImageDecoderRandomCrop(Module):
    def __init__(
        self,
        blob_name: str,
        color_space: str = "BGR",
        num_attempts: int = 10,
        random_seed: Optional[int] = None,
        random_area: Sequence[float] = [0.08, 1.0],
        random_aspect_ratio: Sequence[float] = [0.75, 1.333333],
    ):
        super().__init__()
        seed, has_seed = mirrored_gen_random_seed(random_seed)
        self._op = (
            flow.builtin_op("ofrecord_image_decoder_random_crop")
            .Input("in")
            .Output("out")
            .Attr("name", blob_name)
            .Attr("color_space", color_space)
            .Attr("num_attempts", num_attempts)
            .Attr("random_area", random_area)
            .Attr("random_aspect_ratio", random_aspect_ratio)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .Build()
        )

    def forward(self, input):
        res = self._op(input)[0]
        return res


@oneflow_export("nn.OFRecordImageDecoder")
@experimental_api
class OFRecordImageDecoder(Module):
    def __init__(
        self, blob_name: str, color_space: str = "BGR",
    ):
        super().__init__()
        self._op = (
            flow.builtin_op("ofrecord_image_decoder")
            .Input("in")
            .Output("out")
            .Attr("name", blob_name)
            .Attr("color_space", color_space)
            .Build()
        )

    def forward(self, input):
        res = self._op(input)[0]
        return res


class TensorBufferToListOfTensors(Module):
    def __init__(
        self, out_shapes, out_dtypes, out_num: int = 1, dynamic_out: bool = False
    ):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_buffer_to_list_of_tensors_v2")
            .Input("in")
            .Output("out", out_num)
            .Attr("out_shapes", out_shapes)
            .Attr("out_dtypes", out_dtypes)
            .Attr("dynamic_out", dynamic_out)
            .Build()
        )

    def forward(self, input):
        return self._op(input)


@oneflow_export("tensor_buffer_to_list_of_tensors")
@experimental_api
def tensor_buffer_to_list_of_tensors(tensor, out_shapes, out_dtypes):
    return TensorBufferToListOfTensors(
        [list(out_shape) for out_shape in out_shapes], out_dtypes, len(out_shapes)
    )(tensor)


@oneflow_export("nn.image.Resize")
@experimental_api
class ImageResize(Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        # process deprecated params
        deprecated_param_used = False
        if color_space is not None:
            print(
                "WARNING: color_space has been deprecated. Please use channels instead."
            )
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

            self._op = (
                flow.builtin_op("image_resize_keep_aspect_ratio")
                .Input("in")
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
            # TODO(Liang Depeng)
            # scale = flow.tensor_buffer_to_tensor(
            #     scale, dtype=flow.float32, instance_shape=(2,)
            # )
            # new_size = flow.tensor_buffer_to_tensor(
            #     new_size, dtype=flow.int32, instance_shape=(2,)
            # )
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
            self._op = (
                flow.builtin_op("image_resize_to_fixed")
                .Input("in")
                .Output("out")
                .Output("scale")
                .Attr("target_width", target_w)
                .Attr("target_height", target_h)
                .Attr("channels", channels)
                .Attr("data_type", dtype)
                .Attr("interpolation_type", interpolation_type)
                .Build()
            )

    def forward(self, input):
        res = self._op(input)[0]
        return res


@oneflow_export("tmp.RawDecoder")
@experimental_api
def raw_decoder(
    input_record,
    blob_name: str,
    shape: Sequence[int],
    dtype: flow.dtype,
    dim1_varying_length: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
):
    return OfrecordRawDecoder(
        blob_name, shape, dtype, dim1_varying_length, auto_zero_padding, name
    ).forward(input_record)


@oneflow_export("tmp.OfrecordReader")
@experimental_api
def get_ofrecord_handle(
    ofrecord_dir: str,
    batch_size: int = 1,
    data_part_num: int = 1,
    part_name_prefix: str = "part-",
    part_name_suffix_length: int = -1,
    random_shuffle: bool = False,
    shuffle_buffer_size: int = 1024,
    shuffle_after_epoch: bool = False,
    name: Optional[str] = None,
):
    return OfrecordReader(
        ofrecord_dir,
        batch_size,
        data_part_num,
        part_name_prefix,
        part_name_suffix_length,
        random_shuffle,
        shuffle_buffer_size,
        shuffle_after_epoch,
        name,
    )()
