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
        name: Optional[str] = None,
    ):
        super().__init__()

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
            .Build()
        )

    def forward(self):
        res = self._op()[0]
        return res


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
    ).forward()


if __name__ == "__main__":
    flow.enable_eager_execution(True)
    print(flow.eager_execution_enabled())
    flow.env.init()
    record = flow.tmp.OfrecordReader("/dataset/lenet_mnist/data/ofrecord/train")
    print(type(record))

    # image = tmp.RawDecoder(record, "images", (784,), dtype=flow.int32)
    # print(type(image))
