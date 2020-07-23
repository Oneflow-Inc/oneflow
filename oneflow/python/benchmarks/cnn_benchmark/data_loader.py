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
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(image_size, image_size, 3),
        dtype=flow.float,
        codec=flow.data.ImageCodec(
            [flow.data.ImageResizePreprocessor(image_size, image_size)]
        ),
        preprocessors=[
            flow.data.NormByChannelPreprocessor(
                (123.68, 116.78, 103.94), (255.0, 255.0, 255.0)
            )
        ],
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=batch_size,
        data_part_num=data_part_num,
        name="decode",
    )


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
