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

import operator
import unittest
from functools import reduce

import cv2
import numpy as np

import oneflow as flow
import oneflow.unittest


def _read_images_by_cv(image_files):
    images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
    return images


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


def _roundup(x, n):
    return int((x + n - 1) / n) * n


@flow.unittest.skip_unless_1n1d()
class TestImageBatchAlign(flow.unittest.TestCase):
    def test_image_batch_align(test_case):
        image_files = [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
            "/dataset/mscoco_2017/val2017/000000000785.jpg",
            "/dataset/mscoco_2017/val2017/000000001000.jpg",
        ]
        alignment = 16
        images = _read_images_by_cv(image_files)
        image_shape = _get_images_static_shape(images)
        assert len(image_shape) == 4
        aligned_image_shape = [
            image_shape[0],
            _roundup(image_shape[1], alignment),
            _roundup(image_shape[2], alignment),
            image_shape[3],
        ]
        image_batch_aligner = flow.nn.image.batch_align(
            shape=aligned_image_shape[1:], dtype=flow.float, alignment=alignment
        )
        images_np_arr_static = np.zeros(image_shape, dtype=np.float32)
        for (idx, np_arr) in enumerate(images):
            images_np_arr_static[idx, : np_arr.shape[0], : np_arr.shape[1], :] = np_arr
        input = flow.tensor(
            images_np_arr_static, dtype=flow.float, device=flow.device("cpu")
        )
        images_buffer = flow.tensor_to_tensor_buffer(input, instance_dims=3)
        of_aligned_image = image_batch_aligner(images_buffer).numpy()
        test_case.assertTrue(
            np.array_equal(aligned_image_shape, of_aligned_image.shape)
        )
        empty_image_array = np.zeros(aligned_image_shape, np.float32)
        for (empty_image, image) in zip(empty_image_array, images):
            empty_image[0 : image.shape[0], 0 : image.shape[1], :] = image
        test_case.assertTrue(np.array_equal(of_aligned_image, empty_image_array))


if __name__ == "__main__":
    unittest.main()
