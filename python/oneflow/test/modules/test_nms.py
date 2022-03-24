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
from collections import OrderedDict

import numpy as np
import oneflow as flow
from oneflow.test_utils.test_util import GenArgList


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union_np(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=np.inf)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, np.newaxis] + area2 - inter

    return inter, union


def box_iou_np(boxes1, boxes2):
    inter, union = _box_inter_union_np(boxes1, boxes2)
    iou = inter / union
    return iou


def nms_np(boxes, scores, iou_threshold):
    picked = []
    indexes = np.argsort(-scores)
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = np.squeeze(box_iou_np(rest_boxes, current_box[np.newaxis]), axis=1)
        indexes = indexes[iou <= iou_threshold]

    return np.asarray(picked)


def create_tensors_with_iou(N, iou_thresh):
    boxes = np.random.rand(N, 4) * 100
    boxes[:, 2:] += boxes[:, :2]
    boxes[-1, :] = boxes[0, :]
    x0, y0, x1, y1 = boxes[-1].tolist()
    iou_thresh += 1e-5
    boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
    # Avoid score lists have the same score which will
    # result in an unstable sort.
    scores = np.random.choice(N, N, replace=False)
    return boxes, scores


def _test_nms(test_case, device):
    iou = 0.5
    boxes, scores = create_tensors_with_iou(1000, iou)
    boxes = flow.tensor(boxes, dtype=flow.float32, device=flow.device(device))
    scores = flow.tensor(scores, dtype=flow.float32, device=flow.device(device))
    keep_np = nms_np(boxes.numpy(), scores.numpy(), iou)
    keep = flow.nms(boxes, scores, iou)
    test_case.assertTrue(np.allclose(keep.numpy(), keep_np))


@flow.unittest.skip_unless_1n1d()
class TestNMS(flow.unittest.TestCase):
    def test_nms(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_nms]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
