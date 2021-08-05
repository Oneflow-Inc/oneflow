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
import unittest
import os

import oneflow as flow
import oneflow.unittest


class OFRecordDataLoader(flow.nn.Module):
    def __init__(self, device=None, placement=None, sbp=None):
        super().__init__()
        batch_size = 4
        # don't shuffle, for comparing
        shuffle = False

        self.ofrecord_reader = flow.nn.OfrecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=shuffle,
            shuffle_after_epoch=shuffle,
            device=device,
            placement=placement,
            sbp=sbp,
        )

        self.record_label_decoder = flow.nn.OfrecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )

        self.record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space="RGB"
        )

        self.resize = flow.nn.image.Resize(target_size=[224, 224], dtype=flow.float32)

    def forward(self):
        record = self.ofrecord_reader()
        label = self.record_label_decoder(record)
        image_raw_buffer = self.record_image_decoder(record)
        image = self.resize(image_raw_buffer)[0]
        return image, label


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class DistributedOFRecordReaderTestCase(oneflow.unittest.TestCase):
    def test(test_case):
        rank = flow.distributed.get_rank()
        print(f"DistributedOFRecordReaderTestCase.test on rank {rank}")
        eager_ofrecord_loader = OFRecordDataLoader(device=flow.device("cpu", rank))
        image, label = eager_ofrecord_loader()

        print(f"image: {image.shape}, {image.dtype}")
        # print(image.numpy())
        print(f"label: {label.shape}, {label.dtype}")
        print(label.numpy())

        # class GraphReader(flow.nn.Graph):
        #     def __init__(self):
        #         super().__init__()
        #         self.my_reader = cc_reader

        #     def build(self):
        #         return self.my_reader()

        # reader_g = GraphReader()
        # image, label = reader_g()


if __name__ == "__main__":
    unittest.main()
