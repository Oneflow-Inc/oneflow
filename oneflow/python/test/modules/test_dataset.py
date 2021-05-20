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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestOFRecordModule(flow.unittest.TestCase):
    def test_record(test_case):
        flow.InitEagerGlobalSession()

        batch_size = 1
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

        val_record = record_reader()
        label = record_label_decoder(val_record)
        image_raw_buffer = record_image_decoder(val_record)
        image = resize(image_raw_buffer)
        image = crop_mirror_normal(image)

        # recover image
        image_np = image.numpy()
        image_np = np.squeeze(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * rgb_std + rgb_mean
        image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.uint8)

        # read gt
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_val_image.png")

        test_case.assertEqual(label.numpy()[0], 5)
        test_case.assertTrue(np.array_equal(image_np, gt_np))


if __name__ == "__main__":
    unittest.main()
