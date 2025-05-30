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
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


def nms_op(boxes, scores, iou_threshold: float):
    if boxes.device == flow.device("npu"):
        if boxes.ndim == 2 and boxes.shape[-1] == 4:
            boxes = boxes.unsqueeze(0)
        elif boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(
                f"boxes must be of shape [B, N, 4] or [N, 4], but got {boxes.shape}"
            )

        if scores.ndim == 1:
            scores = scores.unsqueeze(0).unsqueeze(0)
        elif scores.ndim == 2:
            scores = scores.unsqueeze(0)
        elif scores.ndim != 3:
            raise ValueError(
                f"scores must be of shape [B, C, N], [C, N] or [N], but got {scores.shape}"
            )

        if boxes.shape[0] != scores.shape[0]:
            raise ValueError(
                f"batch_size mismatch: boxes {boxes.shape[0]} vs scores {scores.shape[0]}"
            )
        if boxes.shape[1] != scores.shape[2]:
            raise ValueError(
                f"spatial_dimension mismatch: boxes {boxes.shape[1]} vs scores {scores.shape[2]}"
            )

        return flow._C.nms(boxes, scores, iou_threshold)

    score_inds = flow.argsort(scores, dim=0, descending=True)
    boxes = flow._C.gather(boxes, score_inds, axis=0)
    keep = flow._C.nms(boxes, iou_threshold=iou_threshold)
    print(keep.shape)
    index = flow.squeeze(flow.argwhere(keep), dim=[1])
    print(index.shape)
    return flow._C.gather(score_inds, index, axis=0)
