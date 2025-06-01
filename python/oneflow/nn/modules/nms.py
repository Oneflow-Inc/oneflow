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
    score_inds = flow.argsort(scores, dim=0, descending=True)
    if boxes.device == flow.device("npu"):
        sorted_scores = flow.gather(scores, dim=0, index=score_inds)
        keep = flow._C.nms(boxes, sorted_scores, score_inds.to(flow.int32), iou_threshold=iou_threshold)
    else:
        boxes = flow._C.gather(boxes, score_inds, axis=0)
        keep = flow._C.nms(boxes, iou_threshold=iou_threshold)
    index = flow.squeeze(flow.argwhere(keep), dim=[1])
    return flow._C.gather(score_inds, index, axis=0)
