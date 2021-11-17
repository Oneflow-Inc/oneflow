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
            flow.load("/dataset/imagenette/resnet50_pretrained")
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
            6.823004722595215,
            6.818080902099609,
            6.817478179931641,
            6.820215702056885,
            6.820272445678711,
            6.805415630340576,
            6.812217712402344,
            6.822971343994141,
            6.81321907043457,
            6.812097549438477,
            6.808729648590088,
            6.809578895568848,
            6.810042381286621,
            6.81298303604126,
            6.806015968322754,
            6.809454917907715,
            6.808111190795898,
            6.80530309677124,
            6.808160781860352,
            6.809715747833252,
            6.804327487945557,
            6.801260948181152,
            6.801140785217285,
            6.802030086517334,
            6.802935600280762,
            6.793076992034912,
            6.800511360168457,
            6.7988386154174805,
            6.798485279083252,
            6.802251815795898,
            6.798983573913574,
            6.798493385314941,
            6.796577453613281,
            6.787880897521973,
            6.796964645385742,
            6.783697128295898,
            6.7896833419799805,
            6.786165714263916,
            6.790346145629883,
            6.785680770874023,
            6.782796859741211,
            6.784112930297852,
            6.792185306549072,
            6.780761241912842,
            6.778015613555908,
            6.778000354766846,
            6.789952278137207,
            6.773430824279785,
            6.780228614807129,
            6.774554252624512,
            6.77685546875,
            6.7801337242126465,
            6.767944812774658,
            6.7757134437561035,
            6.772693157196045,
            6.770571231842041,
            6.766884803771973,
            6.762784004211426,
            6.765412330627441,
            6.768856048583984,
            6.769237518310547,
            6.77099609375,
            6.765361785888672,
            6.7630228996276855,
            6.757351875305176,
            6.761430740356445,
            6.757913112640381,
            6.756040096282959,
            6.75714111328125,
            6.752540588378906,
            6.7559967041015625,
            6.759932041168213,
            6.756745338439941,
            6.750467300415039,
            6.750478744506836,
            6.750133514404297,
            6.75436544418335,
            6.744396209716797,
            6.753242492675781,
            6.747480392456055,
            6.744192123413086,
            6.744802474975586,
            6.742746829986572,
            6.7499589920043945,
            6.739953517913818,
            6.739869117736816,
            6.744085311889648,
            6.744339942932129,
            6.741791248321533,
            6.737485885620117,
            6.735355377197266,
            6.7377848625183105,
            6.73032283782959,
            6.734944820404053,
            6.7288079261779785,
            6.737483978271484,
            6.730724334716797,
            6.728422164916992,
            6.723917007446289,
            6.734870910644531,
        ]
        for b in range(100):
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
            test_case.assertTrue(np.allclose(l.item(), gt_of_losses[b], atol=1e-05))


if __name__ == "__main__":
    unittest.main()
