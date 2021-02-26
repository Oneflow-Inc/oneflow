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
import random

import numpy as np
import oneflow as flow
import oneflow.typing as oft


def _of_object_bbox_flip(bbox_list, image_size, flip_code):
    bbox_shape = _get_bbox_static_shape(bbox_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def object_bbox_flip_job(
        bbox_def: oft.ListListNumpy.Placeholder(
            shape=tuple(bbox_shape), dtype=flow.float
        ),
        image_size_def: oft.ListNumpy.Placeholder(
            shape=image_size.shape, dtype=flow.int32
        ),
    ):
        bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
        flip_bbox = flow.object_bbox_flip(bbox_buffer, image_size_def, flip_code)
        return flow.tensor_buffer_to_tensor_list(
            flip_bbox, shape=bbox_shape[1:], dtype=flow.float
        )

    input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
    bbox_tensor = object_bbox_flip_job([input_bbox_list], [image_size]).get()
    return bbox_tensor.numpy_lists()[0]


def _get_bbox_static_shape(bbox_list):
    bbox_shapes = [bbox.shape for bbox in bbox_list]
    bbox_static_shape = np.amax(bbox_shapes, axis=0)
    assert isinstance(
        bbox_static_shape, np.ndarray
    ), "bbox_shapes: {}, bbox_static_shape: {}".format(
        str(bbox_shapes), str(bbox_static_shape)
    )
    bbox_static_shape = bbox_static_shape.tolist()
    bbox_static_shape.insert(0, len(bbox_list))
    return bbox_static_shape


def _compare_bbox_flip(
    test_case, anno_file, batch_size, flip_code, print_debug_info=False
):
    from pycocotools.coco import COCO

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()

    bbox_list = []
    image_size_list = []
    sample_cnt = 0
    while sample_cnt < batch_size:
        rand_img_id = random.choice(img_ids)
        anno_ids = coco.getAnnIds(imgIds=[rand_img_id])
        if len(anno_ids) == 0:
            continue
        bbox_array = np.array(
            [coco.anns[anno_id]["bbox"] for anno_id in anno_ids], dtype=np.single
        )
        bbox_list.append(bbox_array)
        image_size_list.append(
            [coco.imgs[rand_img_id]["width"], coco.imgs[rand_img_id]["height"]]
        )
        sample_cnt += 1

    image_size_array = np.array(image_size_list, dtype=np.int32)
    of_bbox_list = _of_object_bbox_flip(bbox_list, image_size_array, flip_code)
    for of_bbox, bbox, image_size in zip(of_bbox_list, bbox_list, image_size_list):
        w, h = image_size
        if flip_code == 1:
            xmin = bbox[:, 0].copy()
            xmax = bbox[:, 2].copy()
            bbox[:, 0] = w - xmax - 1
            bbox[:, 2] = w - xmin - 1
        else:
            raise NotImplementedError

        if print_debug_info:
            print("-" * 20)
            print("ret_bbox:\n", of_bbox.squeeze())
            print("bbox:\n", bbox)

        test_case.assertTrue(np.allclose(of_bbox.squeeze(), bbox))


@flow.unittest.skip_unless_1n1d()
class TestObjectBboxFlip(flow.unittest.TestCase):
    def test_object_bbox_flip(test_case):
        _compare_bbox_flip(
            test_case, "/dataset/mscoco_2017/annotations/instances_val2017.json", 4, 1
        )


if __name__ == "__main__":
    unittest.main()
