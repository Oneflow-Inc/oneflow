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
from oneflow.experimental.load_mnist import load_mnist
from oneflow.ops.user_data_ops import OFRecordBytesDecoder
from oneflow.ops.user_data_ops import OFRecordBytesDecoder as ofrecord_bytes_decoder
from oneflow.ops.user_data_ops import OFRecordImageDecoder
from oneflow.ops.user_data_ops import OFRecordImageDecoder as ofrecord_image_decoder
from oneflow.ops.user_data_ops import OFRecordRawDecoder
from oneflow.ops.user_data_ops import OFRecordRawDecoder as ofrecord_raw_decoder
from oneflow.ops.user_data_ops import OneRecDecoder
from oneflow.ops.user_data_ops import OneRecDecoder as onerec_decoder
from oneflow.ops.user_data_ops import api_coco_reader as coco_reader
from oneflow.ops.user_data_ops import (
    api_ofrecord_image_decoder_random_crop as OFRecordImageDecoderRandomCrop,
)
from oneflow.ops.user_data_ops import (
    api_ofrecord_image_decoder_random_crop as ofrecord_image_decoder_random_crop,
)
from oneflow.ops.user_data_ops import gpt_data_loader as MegatronGPTMMapDataLoader
from oneflow.ops.user_data_ops import gpt_data_loader as megatron_gpt_mmap_data_loader
from oneflow.ops.user_data_ops import ofrecord_image_classification_reader
