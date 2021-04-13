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

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)
from oneflow.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from typing import Optional, List, Tuple, Sequence


@oneflow_export("nn.OfrecordReader")
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
        seed, has_seed = flow.random.gen_seed(random_seed)
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
class CoinFlip(Module):
    def __init__(
        self,
        batch_size: int = 1,
        random_seed: Optional[int] = None,
        probability: float = 0.5,
    ):
        super().__init__()
        seed, has_seed = flow.random.gen_seed(random_seed)
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

    def forward(self, input, mirror):
        res = self._op(input, mirror)[0]
        return res


@oneflow_export("tmp.RawDecoder")
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
