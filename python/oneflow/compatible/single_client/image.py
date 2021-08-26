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
from oneflow.compatible.single_client.ops.user_data_ops import CropMirrorNormalize
from oneflow.compatible.single_client.ops.user_data_ops import (
    CropMirrorNormalize as crop_mirror_normalize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_random_crop as random_crop,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_resize as Resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_resize as resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    api_image_target_resize as target_resize,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_batch_align as batch_align,
)
from oneflow.compatible.single_client.ops.user_data_ops import image_decode as decode
from oneflow.compatible.single_client.ops.user_data_ops import image_flip as flip
from oneflow.compatible.single_client.ops.user_data_ops import (
    image_normalize as normalize,
)
