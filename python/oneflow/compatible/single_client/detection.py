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
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_bbox_flip,
    object_bbox_scale,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_flip as object_segmentation_polygon_flip,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_scale as object_segmentation_polygon_scale,
)
from oneflow.compatible.single_client.ops.user_data_ops import (
    object_segm_poly_to_mask as object_segmentation_polygon_to_mask,
)
