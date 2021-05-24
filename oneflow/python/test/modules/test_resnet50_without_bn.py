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

import oneflow.experimental as flow
from resnet50_model import resnet50, FakeBN


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestResNet50(flow.unittest.TestCase):
    def test_resnet50_without_batchnorm(test_case):
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

        res50_module = resnet50(
            replace_stride_with_dilation=[False, False, False], norm_layer=FakeBN,
        )
        res50_module.train()
        res50_module.load_state_dict(flow.load("/dataset/imagenette/resnet50_models"))
        of_corss_entropy = flow.nn.CrossEntropyLoss()
        res50_module.to("cuda")
        of_corss_entropy.to("cuda")

        learning_rate = 1e-3
        mom = 0.9
        of_sgd = flow.optim.SGD(
            res50_module.parameters(), lr=learning_rate, momentum=mom
        )

        gt_of_losses = [
            46.051700592041016,
            40.29523849487305,
            6.907228946685791,
            6.908342361450195,
            6.90944766998291,
            6.896576881408691,
            6.901367664337158,
            6.9103851318359375,
            6.902885437011719,
            6.9025163650512695,
            6.89929723739624,
            6.900156021118164,
            6.900783538818359,
            6.902735233306885,
            6.897723197937012,
            6.898225784301758,
            6.899064540863037,
            6.896049499511719,
            6.898524761199951,
            6.898486614227295,
            6.894113063812256,
            6.89253044128418,
            6.892385482788086,
            6.893096923828125,
            6.89231538772583,
            6.885106086730957,
            6.890092849731445,
            6.889172554016113,
            6.888751983642578,
            6.891580581665039,
            6.8889079093933105,
            6.889250755310059,
            6.88698148727417,
            6.877960205078125,
            6.886369228363037,
            6.875724792480469,
            6.88068151473999,
            6.876683235168457,
            6.880986213684082,
            6.876288414001465,
            6.873709201812744,
            6.875205993652344,
            6.880834102630615,
            6.871561050415039,
            6.868767261505127,
            6.86946439743042,
            6.877771377563477,
            6.864743709564209,
            6.869909286499023,
            6.8649821281433105,
            6.86630916595459,
            6.869441509246826,
            6.857909202575684,
            6.865201950073242,
            6.862859725952148,
            6.859868049621582,
            6.857898712158203,
            6.854402542114258,
            6.855825424194336,
            6.85800838470459,
            6.858725547790527,
            6.859932899475098,
            6.855479717254639,
            6.852761268615723,
            6.848263740539551,
            6.852200508117676,
            6.848052978515625,
            6.8459272384643555,
            6.847272872924805,
            6.843410968780518,
            6.846282482147217,
            6.848758697509766,
            6.845550060272217,
            6.841163635253906,
            6.840295791625977,
            6.841281414031982,
            6.843825340270996,
            6.835814476013184,
            6.842962265014648,
            6.836810111999512,
            6.8349995613098145,
            6.835197448730469,
            6.832708358764648,
            6.839014530181885,
            6.830626487731934,
            6.829705238342285,
            6.833293914794922,
            6.833970069885254,
            6.831326484680176,
            6.827569007873535,
            6.82619571685791,
            6.8270649909973145,
            6.821228981018066,
            6.824530601501465,
            6.819887161254883,
            6.826042652130127,
            6.8209028244018555,
            6.818319320678711,
            6.814948081970215,
            6.823676109313965,
        ]

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
            test_case.assertEqual(l.item(), gt_of_losses[b])


if __name__ == "__main__":
    unittest.main()
