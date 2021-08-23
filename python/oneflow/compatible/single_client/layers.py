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
from oneflow.compatible.single_client.ops.categorical_ordinal_encode_op import (
    categorical_ordinal_encoder,
)
from oneflow.compatible.single_client.ops.layers import (
    batch_normalization,
    batch_normalization_add_relu,
    batch_normalization_relu,
    conv1d,
    conv2d,
    conv3d,
    dense,
    layer_norm,
    layer_norm_grad,
    layer_norm_param_grad,
)
from oneflow.compatible.single_client.ops.layers import upsample as upsample_2d
from oneflow.compatible.single_client.ops.prelu import prelu
