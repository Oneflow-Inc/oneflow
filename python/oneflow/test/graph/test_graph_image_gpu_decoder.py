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
import os
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


class OFRecordDataLoader(flow.nn.Module):
    def __init__(self):
        super().__init__()
        batch_size = 4
        image_size = 224
        self.train_record_reader = flow.nn.OFRecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=True,
            shuffle_after_epoch=True,
            # placement=flow.placement("cpu", ranks=[0]),
            # sbp=[flow.sbp.broadcast]
        )

        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )
        self.bytes_decoder = flow.nn.OFRecordBytesDecoder("encoded")
        self.image_gpu_decoder = flow.nn.OFRecordImageGpuDecoderRandomCropResize(
            target_width=image_size, target_height=image_size, num_workers=3
        )

        color_space = "RGB"
        output_layout = "NHWC"

        self.flip = flow.nn.CoinFlip(
            batch_size=batch_size,
            # placement=flow.placement("cpu", ranks=[0]),
            # sbp=[flow.sbp.broadcast]
        )

        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        self.crop_mirror_norm = flow.nn.CropMirrorNormalize(
            color_space=color_space,
            output_layout=output_layout,
            mean=rgb_mean,
            std=rgb_std,
            output_dtype=flow.float,
        )

    def forward(self) -> (flow.Tensor, flow.Tensor):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        encoded = self.bytes_decoder(train_record)
        image = self.image_gpu_decoder(encoded)
        rng = self.flip()
        if image.is_cuda:
            rng = rng.to("cuda")
        image = self.crop_mirror_norm(image, rng)
        return image, label


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestImageGpuDecoderGraph(oneflow.unittest.TestCase):
    def test_image_gpu_decoder_graph(test_case):
        cc_reader = OFRecordDataLoader()

        class GraphReader(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.my_reader = cc_reader

            def build(self):
                return self.my_reader()

        reader_g = GraphReader()
        image, label = reader_g()
        print(image.shape)
        print(label)


if __name__ == "__main__":
    unittest.main()
