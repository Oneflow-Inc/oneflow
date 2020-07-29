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
from __future__ import absolute_import, division, print_function

import oneflow as flow


def load_imagenet(data_dir, image_size, batch_size, data_part_num):
    rgb_mean = [123.68, 116.78, 103.94]
    rgb_std = [255.0, 255.0, 255.0]
    ofrecord = flow.data.ofrecord_reader(
        data_dir, batch_size=batch_size, data_part_num=data_part_num, name="decode",
    )
    image = flow.data.ofrecord_image_decoder(ofrecord, "encoded", color_space="RGB")
    label = flow.data.ofrecord_raw_decoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32
    )
    rsz = flow.image.resize(
        image, resize_x=image_size, resize_y=image_size, color_space="RGB"
    )
    normal = flow.image.crop_mirror_normalize(
        rsz,
        color_space="RGB",
        output_layout="NCHW",
        mean=rgb_mean,
        std=rgb_std,
        output_dtype=flow.float,
    )
    return label, normal


def load_synthetic(image_size, batch_size):
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(image_size, image_size, 3), dtype=flow.float, batch_size=batch_size
    )

    return label, image
