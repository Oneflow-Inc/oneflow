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
from resnet50_model import FakeBN, resnet50

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestResNet50(flow.unittest.TestCase):
    def test_resnet50_without_batchnorm(test_case):
        batch_size = 32
        color_space = "RGB"
        height = 224
        width = 224
        output_layout = "NCHW"
        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        record_reader = flow.nn.OFRecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            shuffle_after_epoch=False,
        )
        record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        record_label_decoder = flow.nn.OFRecordRawDecoder(
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
        res50_module = resnet50(
            replace_stride_with_dilation=[False, False, False], norm_layer=FakeBN
        )
        res50_module.train()
        res50_module.load_state_dict(
            flow.load("/dataset/resnet50_wo_bn_weights_for_ci")
        )
        of_corss_entropy = flow.nn.CrossEntropyLoss()
        res50_module.to("cuda")
        of_corss_entropy.to("cuda")
        learning_rate = 0.001
        mom = 0.9
        of_sgd = flow.optim.SGD(
            res50_module.parameters(), lr=learning_rate, momentum=mom
        )
        gt_of_losses = [
            49.83235168457031,
            36.34172821044922,
            23.585250854492188,
            15.628865242004395,
            9.552209854125977,
            8.11514663696289,
            6.364114284515381,
            6.442500114440918,
            4.439807891845703,
            4.024901866912842,
            4.7038373947143555,
            4.253284454345703,
            4.5806169509887695,
            4.158677577972412,
            3.0066077709198,
            4.611920356750488,
            4.46696138381958,
            2.9725658893585205,
            3.2383458614349365,
            3.605447292327881,
            3.8676259517669678,
            3.2477705478668213,
            2.9191272258758545,
            3.162745475769043,
            3.0127673149108887,
            2.615905284881592,
            2.7866411209106445,
            3.471228837966919,
            2.9467897415161133,
            3.3623316287994385,
        ]
        for b in range(len(gt_of_losses)):
            val_record = record_reader()
            label = record_label_decoder(val_record)
            image_raw_buffer = record_image_decoder(val_record)
            image = resize(image_raw_buffer)[0]
            image = crop_mirror_normal(image)
            image = image.to("cuda")
            label = label.to("cuda")
            logits = res50_module(image)
            loss = of_corss_entropy(logits, label)
            loss.backward()
            of_sgd.step()
            of_sgd.zero_grad()
            l = loss.numpy()
            test_case.assertTrue(
                np.allclose(l.item(), gt_of_losses[b], rtol=1e-2, atol=1e-3)
            )


if __name__ == "__main__":
    unittest.main()
