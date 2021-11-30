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
import random
import sys
import traceback
from typing import List, Optional, Sequence, Tuple, Union

import oneflow as flow
from oneflow.framework.tensor import Tensor, TensorTuple
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple
import oneflow.framework.id_util as id_util


def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True
    return (seed, has_seed)


class OFRecordReader(Module):
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
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
        name: Optional[str] = None,
    ):
        super().__init__()

        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")

        nd_sbp = []

        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            assert device is None

        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(sbp, flow.sbp.sbp):
                nd_sbp.append(sbp._ToAttrStr())
                sbp = (sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
                    nd_sbp.append(elem._ToAttrStr())
            assert len(nd_sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp

        self.sbp = sbp

        (seed, has_seed) = mirrored_gen_random_seed(random_seed)
        self._op = (
            flow.builtin_op("OFRecordReader")
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
            .Attr("nd_sbp", nd_sbp)
            .Build()
        )
        self.attrs = flow._oneflow_internal.MutableCfgAttrMap()

    def forward(self):
        if self.placement is not None:
            res = self._op.apply(self.placement, self.sbp, self.attrs)[0]
        else:
            res = self._op.apply(self.device, self.attrs)[0]
        return res


class OFRecordRawDecoder(Module):
    def __init__(
        self,
        blob_name: str,
        shape: Sequence[int],
        dtype: flow.dtype,
        dim1_varying_length: bool = False,
        truncate: bool = False,
        auto_zero_padding: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        if auto_zero_padding:
            print(
                "WARNING: auto_zero_padding has been deprecated, Please use truncate instead.\n"
            )
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        self._op = (
            flow.builtin_op("ofrecord_raw_decoder")
            .Input("in")
            .Output("out")
            .Attr("name", blob_name)
            .Attr("shape", shape)
            .Attr("data_type", dtype)
            .Attr("dim1_varying_length", dim1_varying_length)
            .Attr("truncate", truncate or auto_zero_padding)
            .Build()
        )

    def forward(self, input):
        res = self._op(input)[0]
        return res


class CoinFlip(Module):
    def __init__(
        self,
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        probability: float = 0.5,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        nd_sbp = []

        self.placement = placement
        if placement is None:
            if device is None:
                self.device = flow.device("cpu")
        else:
            assert device is None

        if placement is not None:
            assert isinstance(sbp, (flow.sbp.sbp, tuple, list)), "sbp: %s" % sbp
            if isinstance(sbp, flow.sbp.sbp):
                nd_sbp.append(sbp._ToAttrStr())
                sbp = (sbp,)
            else:
                for elem in sbp:
                    assert isinstance(elem, flow.sbp.sbp), "sbp: %s" % sbp
                    nd_sbp.append(elem._ToAttrStr())
            assert len(nd_sbp) == len(placement.hierarchy)
        else:
            assert sbp is None, "sbp: %s" % sbp

        self.sbp = sbp

        (seed, has_seed) = mirrored_gen_random_seed(random_seed)

        self._op = (
            flow.builtin_op("coin_flip")
            .Output("out")
            .Attr("batch_size", batch_size)
            .Attr("probability", probability)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .Attr("nd_sbp", nd_sbp)
            .Build()
        )
        self.attrs = flow._oneflow_internal.MutableCfgAttrMap()

    def forward(self):
        if self.placement is not None:
            res = self._op.apply(self.placement, self.sbp, self.attrs)[0]
        else:
            res = self._op.apply(self.device, self.attrs)[0]
        return res


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
        self._op_uint8_with_mirror = (
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
        self._op_uint8_no_mirror = (
            flow.builtin_op("crop_mirror_normalize_from_uint8")
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
        self._op_buffer_with_mirror = (
            flow.builtin_op("crop_mirror_normalize_from_tensorbuffer")
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

        self._op_buffer_no_mirror = (
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
        if mirror is not None:
            if input.dtype is flow.uint8:
                res = self._op_uint8_with_mirror(input, mirror)[0]
            elif input.dtype is flow.tensor_buffer:
                res = self._op_buffer_with_mirror(input, mirror)[0]
            else:
                print(
                    "ERROR! oneflow.nn.CropMirrorNormalize module NOT support input dtype = ",
                    input.dtype,
                )
                raise NotImplementedError
        else:
            if input.dtype is flow.uint8:
                res = self._op_uint8_no_mirror(input)[0]
            elif input.dtype is flow.tensor_buffer:
                res = self._op_buffer_no_mirror(input)[0]
            else:
                print(
                    "ERROR! oneflow.nn.CropMirrorNormalize module NOT support input dtype = ",
                    input.dtype,
                )
                raise NotImplementedError
        return res


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
        (seed, has_seed) = mirrored_gen_random_seed(random_seed)
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


class OFRecordImageDecoder(Module):
    def __init__(self, blob_name: str, color_space: str = "BGR"):
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


class OFRecordImageGpuDecoderRandomCropResize(Module):
    def __init__(
        self,
        target_width: int,
        target_height: int,
        num_attempts: Optional[int] = None,
        seed: Optional[int] = None,
        random_area: Optional[Sequence[float]] = None,
        random_aspect_ratio: Optional[Sequence[float]] = None,
        num_workers: Optional[int] = None,
        warmup_size: Optional[int] = None,
        max_num_pixels: Optional[int] = None,
    ):
        super().__init__()
        gpu_decoder_conf = (
            flow._oneflow_internal.oneflow.core.operator.op_conf.ImageDecoderRandomCropResizeOpConf()
        )
        gpu_decoder_conf.set_in("error_input_need_to_be_replaced")
        gpu_decoder_conf.set_out("out")
        gpu_decoder_conf.set_target_width(target_width)
        gpu_decoder_conf.set_target_height(target_height)
        if num_attempts is not None:
            gpu_decoder_conf.set_num_attempts(num_attempts)
        if seed is not None:
            gpu_decoder_conf.set_seed(seed)
        if random_area is not None:
            assert len(random_area) == 2
            gpu_decoder_conf.set_random_area_min(random_area[0])
            gpu_decoder_conf.set_random_area_max(random_area[1])
        if random_aspect_ratio is not None:
            assert len(random_aspect_ratio) == 2
            gpu_decoder_conf.set_random_aspect_ratio_min(random_aspect_ratio[0])
            gpu_decoder_conf.set_random_aspect_ratio_max(random_aspect_ratio[1])
        if num_workers is not None:
            gpu_decoder_conf.set_num_workers(num_workers)
        if warmup_size is not None:
            gpu_decoder_conf.set_warmup_size(warmup_size)
        if max_num_pixels is not None:
            gpu_decoder_conf.set_max_num_pixels(max_num_pixels)

        self._op = flow._oneflow_internal.one.ImageDecoderRandomCropResizeOpExpr(
            id_util.UniqueStr("ImageGpuDecoder"), gpu_decoder_conf, ["in"], ["out"]
        )
        self.attrs = flow._oneflow_internal.MutableCfgAttrMap()

    def forward(self, input):
        if not input.is_lazy:
            print(
                "ERROR! oneflow.nn.OFRecordImageGpuDecoderRandomCropResize module ",
                "NOT support run as eager module, please use it in nn.Graph.",
            )
            raise NotImplementedError
        res = self._op.apply([input], self.attrs)[0]
        if not res.is_cuda:
            print(
                "WARNING! oneflow.nn.OFRecordImageGpuDecoderRandomCropResize ONLY support ",
                "CUDA runtime version >= 10.2, so now it degenerates into CPU decode version.",
            )
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


def tensor_buffer_to_list_of_tensors(tensor, out_shapes, out_dtypes):
    return TensorBufferToListOfTensors(
        [list(out_shape) for out_shape in out_shapes], out_dtypes, len(out_shapes)
    )(tensor)


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
        color_space: Optional[str] = None,
        interp_type: Optional[str] = None,
        resize_shorter: int = 0,
        resize_x: int = 0,
        resize_y: int = 0,
    ):
        super().__init__()
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
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
        else:
            if (
                not isinstance(target_size, (list, tuple))
                or len(target_size) != 2
                or (not all((isinstance(size, int) for size in target_size)))
            ):
                raise ValueError(
                    "target_size must be a form like (width, height) when keep_aspect_ratio is False"
                )
            if dtype is None:
                dtype = flow.uint8
            (target_w, target_h) = target_size
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
        res = self._op(input)
        res_image = res[0]
        if len(res) == 3:
            new_size = flow.tensor_buffer_to_tensor(
                res[1], dtype=flow.int32, instance_shape=(2,)
            )
            scale = flow.tensor_buffer_to_tensor(
                res[2], dtype=flow.float32, instance_shape=(2,)
            )
        else:
            new_size = None
            scale = res[1]
        return (res_image, scale, new_size)


def raw_decoder(
    input_record,
    blob_name: str,
    shape: Sequence[int],
    dtype: flow.dtype,
    dim1_varying_length: bool = False,
    truncate: bool = False,
    auto_zero_padding: bool = False,
    name: Optional[str] = None,
):
    if auto_zero_padding:
        print(
            "WARNING: auto_zero_padding has been deprecated, Please use truncate instead.\n            "
        )
    return OFRecordRawDecoder(
        blob_name,
        shape,
        dtype,
        dim1_varying_length,
        truncate or auto_zero_padding,
        name,
    ).forward(input_record)


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
    return OFRecordReader(
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


class ImageFlip(Module):
    """This operator flips the images.

    The flip code corresponds to the different flip mode:

    0 (0x00): Non Flip

    1 (0x01): Horizontal Flip

    2 (0x02): Vertical Flip

    3 (0x03): Both Horizontal and Vertical Flip

    Args:
        images: The input images.
        flip_code: The flip code.

    Returns:
        The result image.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> arr = np.array([
        ...    [[[1, 2, 3], [3, 2, 1]],
        ...     [[2, 3, 4], [4, 3, 2]]],
        ...    [[[3, 4, 5], [5, 4, 3]],
        ...     [[4, 5, 6], [6, 5, 4]]]])
        >>> image_tensors = flow.Tensor(arr, device=flow.device("cpu"))
        >>> image_tensor_buffer = flow.tensor_to_tensor_buffer(image_tensors, instance_dims=3)
        >>> flip_code = flow.ones(arr.shape[0], dtype=flow.int8)
        >>> output = nn.image.flip()(image_tensor_buffer, flip_code).numpy()
        >>> output[0]
        array([[[3., 2., 1.],
                [1., 2., 3.]],
        <BLANKLINE>
               [[4., 3., 2.],
                [2., 3., 4.]]], dtype=float32)
        >>> output[1]
        array([[[5., 4., 3.],
                [3., 4., 5.]],
        <BLANKLINE>
               [[6., 5., 4.],
                [4., 5., 6.]]], dtype=float32)
    """

    def __init__(self):
        super().__init__()

    def forward(self, images, flip_code):
        return flow._C.image_flip(images, flip_code=flip_code)


class ImageDecode(Module):
    def __init__(self, dtype: flow.dtype = flow.uint8, color_space: str = "BGR"):
        super().__init__()
        self._op = (
            flow.builtin_op("image_decode")
            .Input("in")
            .Output("out")
            .Attr("color_space", color_space)
            .Attr("data_type", dtype)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class ImageNormalize(Module):
    def __init__(self, std: Sequence[float], mean: Sequence[float]):
        super().__init__()
        self._op = (
            flow.builtin_op("image_normalize")
            .Input("in")
            .Output("out")
            .Attr("std", std)
            .Attr("mean", mean)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class COCOReader(Module):
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
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)

        nd_sbp = []
        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            if device is not None:
                raise ValueError(
                    "when param sbp is specified, param device should not be specified"
                )

            if isinstance(sbp, (tuple, list)):
                for sbp_item in sbp:
                    if not isinstance(sbp_item, flow.sbp.sbp):
                        raise ValueError(f"invalid sbp item: {sbp_item}")
                    nd_sbp.append(sbp_item._ToAttrStr())
            elif isinstance(sbp, flow.sbp.sbp):
                nd_sbp.append(sbp._ToAttrStr())
                sbp = (sbp,)
            else:
                raise ValueError(f"invalid param sbp: {sbp}")

            if len(nd_sbp) != len(placement.hierarchy):
                raise ValueError(
                    "dimensions of sbp and dimensions of hierarchy of placement don't equal"
                )
        self.sbp = sbp

        self._op = (
            flow.builtin_op("COCOReader")
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
            .Attr("nd_sbp", nd_sbp)
            .Build()
        )
        self.attrs = flow._oneflow_internal.MutableCfgAttrMap()

    def forward(self):
        if self.placement is None:
            # local apply
            outputs = self._op.apply(self.device, self.attrs)
        else:
            # consistent apply
            outputs = self._op.apply(self.placement, self.sbp, self.attrs)

        # COCOReader has multiple output, so it return a TensorTuple
        # convert TensorTuple to tuple of Tensor
        assert isinstance(outputs, TensorTuple)
        ret = tuple(out for out in outputs)
        return ret


class ImageBatchAlign(Module):
    def __init__(self, shape: Sequence[int], dtype: flow.dtype, alignment: int):
        super().__init__()
        self._op = (
            flow.builtin_op("image_batch_align")
            .Input("in")
            .Output("out")
            .Attr("shape", shape)
            .Attr("data_type", dtype)
            .Attr("alignment", alignment)
            .Attr("dynamic_out", False)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class OFRecordBytesDecoder(Module):
    r"""This operator reads an tensor as bytes. The output might need

    further decoding process like cv2.imdecode() for images and decode("utf-8")

    for characters,depending on the downstream task.

    Args:
        blob_name: The name of the target feature in OFRecord.

        name: The name for this component in the graph.

        input: the Tensor which might be provided by an OFRecordReader.

    Returns:

        The result Tensor encoded with bytes.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> def example():
        ...      batch_size = 16
        ...      record_reader = flow.nn.OFRecordReader(
        ...         "dataset/",
        ...         batch_size=batch_size,
        ...         part_name_suffix_length=5,
        ...      )
        ...      val_record = record_reader()

        ...      bytesdecoder_img = flow.nn.OFRecordBytesDecoder("encoded")

        ...      image_bytes_batch = bytesdecoder_img(val_record)

        ...      image_bytes = image_bytes_batch.numpy()[0]
        ...      return image_bytes
        ... example()  # doctest: +SKIP
        array([255 216 255 ...  79 255 217], dtype=uint8)



    """

    def __init__(self, blob_name: str, name: Optional[str] = None):
        super().__init__()
        if name is not None:
            print("WARNING: name has been deprecated and has NO effect.\n")
        self._op = (
            flow.builtin_op("ofrecord_bytes_decoder")
            .Input("in")
            .Output("out")
            .Attr("name", blob_name)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class GPTIndexedBinDataReader(Module):
    def __init__(
        self,
        data_file_prefix: str,
        seq_length: int,
        num_samples: int,
        batch_size: int,
        dtype: flow.dtype = flow.int64,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
        split_sizes: Optional[Sequence[str]] = None,
        split_index: Optional[int] = None,
        device: Union[flow.device, str] = None,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()

        nd_sbp = []
        self.placement = placement
        if placement is None:
            self.device = device or flow.device("cpu")
        else:
            if device is not None:
                raise ValueError(
                    "when param sbp is specified, param device should not be specified"
                )

            if isinstance(sbp, (tuple, list)):
                for sbp_item in sbp:
                    if not isinstance(sbp_item, flow.sbp.sbp):
                        raise ValueError(f"invalid sbp item: {sbp_item}")
                    nd_sbp.append(sbp_item._ToAttrStr())
            elif isinstance(sbp, flow.sbp.sbp):
                nd_sbp.append(sbp._ToAttrStr())
                sbp = (sbp,)
            else:
                raise ValueError(f"invalid param sbp: {sbp}")

            if len(nd_sbp) != len(placement.hierarchy):
                raise ValueError(
                    "dimensions of sbp and dimensions of hierarchy of placement don't equal"
                )
        self.sbp = sbp

        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)

        if split_index is None:
            split_index = 0

        if split_sizes is None:
            split_sizes = (1,)

        if split_index >= len(split_sizes):
            raise ValueError(
                "split index {} is out of range, split_sizes {}".formart(
                    split_index, split_sizes
                )
            )

        op_builder = (
            flow.builtin_op("megatron_gpt_mmap_data_loader")
            .Output("out")
            .Attr("data_file_prefix", data_file_prefix)
            .Attr("seq_length", seq_length)
            .Attr("label_length", 1)
            .Attr("num_samples", num_samples)
            .Attr("batch_size", batch_size)
            .Attr("dtype", dtype)
            .Attr("shuffle", shuffle)
            .Attr("random_seed", random_seed)
            .Attr("split_sizes", split_sizes)
            .Attr("split_index", split_index)
            .Attr("nd_sbp", nd_sbp)
        )
        self.op_ = op_builder.Build()
        self.attrs = flow._oneflow_internal.MutableCfgAttrMap()

    def forward(self):
        if self.placement is None:
            output = self.op_.apply(self.device, self.attrs)[0]
        else:
            output = self.op_.apply(self.placement, self.sbp, self.attrs)[0]
        return output


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
