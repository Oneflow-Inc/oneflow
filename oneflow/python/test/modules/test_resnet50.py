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

import cv2
import numpy as np

import oneflow.experimental as flow
from resnet50_model import resnet50


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestResNet50(flow.unittest.TestCase):
    def test_resnet50(test_case):
        # init ofrecord
        flow.InitEagerGlobalSession()

        batch_size = 32
        color_space = "RGB"
        height = 224
        width = 224
        output_layout = "NCHW"
        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        record_reader = flow.nn.OfrecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            shuffle_after_epoch=False,
        )
        record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        record_label_decoder = flow.nn.OfrecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )
        resize = flow.nn.image.Resize(
            resize_side="shorter", keep_aspect_ratio=True, target_size=256
        )
        crop_mirror_normal = flow.nn.CropMirrorNormalize(
            color_space=color_space,
            output_layout=output_layout,
            crop_h=height,
            crop_w=width,
            crop_pos_y=0.5,
            crop_pos_x=0.5,
            mean=rgb_mean,
            std=rgb_std,
            output_dtype=flow.float,
        )

        res50_module = resnet50()
        res50_module.eval()
        res50_module.load_state_dict(flow.load("/dataset/imagenette/resnet50_models"))
        of_corss_entropy = flow.nn.CrossEntropyLoss()
        res50_module.to("cuda")
        of_corss_entropy.to("cuda")

        learning_rate = 0.00001
        mom = 0.9
        of_sgd = flow.optim.SGD(
            res50_module.parameters(), lr=learning_rate, momentum=mom
        )

        of_losses = []
        gt_of_losses = []

        with open("/dataset/imagenette/resnet50_loss.txt", "r") as lines:
            for line in lines:
                arr = line.strip()
                gt_of_losses.append(float(arr))

        errors = 0.0
        for b in range(100):
            val_record = record_reader()
            label = record_label_decoder(val_record)
            image_raw_buffer = record_image_decoder(val_record)
            image = resize(image_raw_buffer)
            image = crop_mirror_normal(image)
            image = image.to("cuda")
            label = label.to("cuda")
            logits = res50_module(image)
            loss = of_corss_entropy(logits, label)
            loss.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            l = loss.numpy()[0]
            of_losses.append(l)
            errors += np.abs(of_losses[b] - gt_of_losses[b])

        test_case.assertTrue((errors / 100) < 1e-3)


if __name__ == "__main__":
    unittest.main()
