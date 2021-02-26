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
import os
import random

import cv2
import numpy as np
import oneflow as flow
import oneflow.typing as oft


def _random_sample_images(anno_file, image_dir, batch_size):
    from pycocotools.coco import COCO

    image_files = []
    image_ids = []
    batch_group_id = -1

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    while len(image_files) < batch_size:
        rand_img_id = random.choice(img_ids)
        img_h = coco.imgs[rand_img_id]["height"]
        img_w = coco.imgs[rand_img_id]["width"]
        group_id = int(img_h / img_w)

        if batch_group_id == -1:
            batch_group_id = group_id

        if group_id != batch_group_id:
            continue

        anno_ids = coco.getAnnIds(imgIds=[rand_img_id])
        if len(anno_ids) == 0:
            continue

        image_files.append(os.path.join(image_dir, coco.imgs[rand_img_id]["file_name"]))
        image_ids.append(rand_img_id)

    assert len(image_files) == len(image_ids)
    images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
    bbox_list = _get_images_bbox_list(coco, image_ids)
    return images, bbox_list


def _get_images_bbox_list(coco, image_ids):
    bbox_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        anno_ids = list(
            filter(lambda anno_id: coco.anns[anno_id]["iscrowd"] == 0, anno_ids)
        )
        bbox_array = np.array(
            [coco.anns[anno_id]["bbox"] for anno_id in anno_ids], dtype=np.single
        )
        bbox_list.append(bbox_array)

    return bbox_list


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0)
    assert isinstance(
        image_static_shape, np.ndarray
    ), "image_shapes: {}, image_static_shape: {}".format(
        str(image_shapes), str(image_static_shape)
    )
    image_static_shape = image_static_shape.tolist()
    image_static_shape.insert(0, len(image_shapes))
    return image_static_shape


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


def _of_target_resize_bbox_scale(images, bbox_list, target_size, max_size):
    image_shape = _get_images_static_shape(images)
    bbox_shape = _get_bbox_static_shape(bbox_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def target_resize_bbox_scale_job(
        image_def: oft.ListListNumpy.Placeholder(
            shape=tuple(image_shape), dtype=flow.float
        ),
        bbox_def: oft.ListListNumpy.Placeholder(
            shape=tuple(bbox_shape), dtype=flow.float
        ),
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
        resized_images_buffer, new_size, scale = flow.image_target_resize(
            images_buffer, target_size=target_size, max_size=max_size
        )
        bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
        scaled_bbox = flow.object_bbox_scale(bbox_buffer, scale)
        scaled_bbox_list = flow.tensor_buffer_to_tensor_list(
            scaled_bbox, shape=bbox_shape[1:], dtype=flow.float
        )
        return scaled_bbox_list, new_size

    input_image_list = [np.expand_dims(image, axis=0) for image in images]
    input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
    output_bbox_list, output_image_size = target_resize_bbox_scale_job(
        [input_image_list], [input_bbox_list]
    ).get()
    return output_bbox_list.numpy_lists()[0], output_image_size.numpy_list()[0]


def _compare_bbox_scale(
    test_case,
    anno_file,
    image_dir,
    batch_size,
    target_size,
    max_size,
    print_debug_info=False,
):
    images, bbox_list = _random_sample_images(anno_file, image_dir, batch_size)
    of_bbox_list, image_size_list = _of_target_resize_bbox_scale(
        images, bbox_list, target_size, max_size
    )

    for image, bbox, of_bbox, image_size in zip(
        images, bbox_list, of_bbox_list, image_size_list
    ):
        w, h = image_size
        oh, ow = image.shape[0:2]
        scale_h = h / oh
        scale_w = w / ow
        bbox[:, 0] *= scale_w
        bbox[:, 1] *= scale_h
        bbox[:, 2] *= scale_w
        bbox[:, 3] *= scale_h
        test_case.assertTrue(np.allclose(bbox, of_bbox))


@flow.unittest.skip_unless_1n1d()
class TestObjectBboxScale(flow.unittest.TestCase):
    def test_object_bbox_scale(test_case):
        _compare_bbox_scale(
            test_case,
            "/dataset/mscoco_2017/annotations/instances_val2017.json",
            "/dataset/mscoco_2017/val2017",
            4,
            800,
            1333,
        )


if __name__ == "__main__":
    unittest.main()
