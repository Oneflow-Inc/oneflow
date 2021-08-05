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
            "/dataset/imagenet_227/train/32",
            batch_size=batch_size,
            data_part_num=2,
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

        self.resize = flow.nn.image.Resize(target_size=[227, 227], dtype=flow.float32)

    def forward(self):
        record = self.ofrecord_reader()
        label = self.record_label_decoder(record)
        image_raw_buffer = self.record_image_decoder(record)
        image = self.resize(image_raw_buffer)[0]
        return image, label


class DataLoaderGraph(flow.nn.Graph):
    def __init__(self, loader):
        super().__init__()
        self.loader_ = loader

    def build(self):
        return self.loader_()


# @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
# @flow.unittest.skip_unless_1n2d()
# class DistributedOFRecordReaderTestCase(oneflow.unittest.TestCase):
def test(test_case=None):
    rank = flow.distributed.get_rank()
    print(f"DistributedOFRecordReaderTestCase.test on rank {rank} {os.getpid()}")

    # eager_ofrecord_loader = OFRecordDataLoader(device=flow.device("cpu", rank))

    lazy_consistent_loader = OFRecordDataLoader(
        placement=flow.placement("cpu", {0: [0]}), sbp=[flow.sbp.split(0)]
    )
    graph = DataLoaderGraph(lazy_consistent_loader)

    iteration = 1
    for i in range(iteration):
        g_image, g_label = graph()
        print(
            f"rank {rank} graph output image: {g_image.shape}, {g_image.dtype}, placement: {g_image.placement}"
            # f"\n{g_image.to_local().numpy().mean()}"
        )
        print(
            f"rank {rank} graph output label: {g_label.shape}, {g_label.dtype}, placement: {g_image.placement}"
            # f"\n{g_label.to_local().numpy()}"
        )

        # image, label = eager_ofrecord_loader()
        # print(
        #     f"rank {rank} image: {image.shape}, {image.dtype}"
        #     f"\n{image.to_local().numpy().mean()}"
        # )
        # print(
        #     f"rank {rank} label: {label.shape}, {label.dtype}"
        #     f"\n{label.to_local().numpy()}"
        # )

        print(f"{'-' * 20} rank {rank} iter {i} complete {'-' * 20}")


if __name__ == "__main__":
    # unittest.main()
    test()
