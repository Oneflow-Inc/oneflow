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
import numpy as np

import oneflow as flow
import oneflow.unittest


class COCODataLoader(flow.nn.Module):
    def __init__(
        self,
        anno_file="/dataset/mscoco_2017/annotations/instances_val2017.json",
        image_dir="/dataset/mscoco_2017/val2017",
        batch_size=2,
        device=None,
        placement=None,
        sbp=None,
    ):
        super().__init__()
        self.coco_reader = flow.nn.COCOReader(
            annotation_file=anno_file,
            image_dir=image_dir,
            batch_size=batch_size,
            shuffle=True,
            random_seed=12345,
            stride_partition=True,
            device=device,
            placement=placement,
            sbp=sbp,
        )
        self.image_decoder = flow.nn.image.decode(dtype=flow.float32)
        self.resize = flow.nn.image.Resize(target_size=[224, 224], dtype=flow.float32)

    def forward(self):
        outputs = self.coco_reader()
        # decode images
        image = self.image_decoder(outputs[0])
        fixed_image = self.resize(image)[0]
        image_id = outputs[1]
        image_size = outputs[2]
        return fixed_image, image_id, image_size


class DataLoaderGraph(flow.nn.Graph):
    def __init__(self, loader):
        super().__init__()
        self.loader_ = loader

    def build(self):
        return self.loader_()


@flow.unittest.skip_unless_1n2d()
class COCODataLoaderDistributedTestCase(oneflow.unittest.TestCase):
    def test_case1(test_case):
        rank = flow.env.get_rank()
        # pid = os.getpid()
        # print(f"[{pid}][{rank}] COCODataLoaderDistributedTestCase.test_case1")

        eager_coco_loader = COCODataLoader(
            batch_size=2, device=flow.device("cpu", rank)
        )

        global_coco_loader = COCODataLoader(
            batch_size=4,
            placement=flow.placement("cpu", ranks=[0, 1]),
            sbp=[flow.sbp.split(0)],
        )
        coco_loader_graph = DataLoaderGraph(global_coco_loader)
        # coco_loader_graph.debug()

        iteration = 1
        for i in range(iteration):
            image, image_id, image_size = eager_coco_loader()

            # print(f"image: {image.numpy().mean()} ")
            # print(f"image_id: {image_id.numpy()}")
            # print(f"image_size: {image_size.numpy()}")

            g_image, g_image_id, g_image_size = coco_loader_graph()

            # print(f"{'-' * 20} rank {rank} iter {i} complete {'-' * 20}")
            test_case.assertTrue(np.allclose(image.numpy(), g_image.to_local().numpy()))
            test_case.assertTrue(
                np.allclose(image_id.numpy(), g_image_id.to_local().numpy())
            )
            test_case.assertTrue(
                np.allclose(image_size.numpy(), g_image_size.to_local().numpy())
            )


if __name__ == "__main__":
    unittest.main()
