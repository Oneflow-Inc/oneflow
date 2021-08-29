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

import os


class OFRecordDataLoader(flow.nn.Module):
    def __init__(
        self,
        ofrecord_root: str = "./ofrecord",
        mode: str = "train",  # "val"
        dataset_size: int = 9469,
        batch_size: int = 1,
    ):
        super().__init__()
        channel_last = False
        output_layout = "NHWC" if channel_last else "NCHW"
        self.train_record_reader = flow.nn.OFRecordReader(
            ofrecord_root,
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
        )
        self.record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 224
        width = 224

        self.record_image_decoder = (
            flow.nn.OFRecordImageDecoderRandomCrop("encoded", color_space=color_space)
            if mode == "train"
            else flow.nn.OFRecordImageDecoder("encoded", color_space=color_space)
        )

        self.resize = (
            flow.nn.image.Resize(target_size=[height, width])
            if mode == "train"
            else flow.nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=256
            )
        )

        self.flip = flow.nn.CoinFlip(batch_size=batch_size) if mode == "train" else None

        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        self.crop_mirror_norm = (
            flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
            if mode == "train"
            else flow.nn.CropMirrorNormalize(
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
        )

        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)
        image = self.resize(image_raw_buffer)[0]
        rng = self.flip() if self.flip != None else None
        image = self.crop_mirror_norm(image, rng)

        return image, label
