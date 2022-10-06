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

import cv2
import numpy as np

import oneflow as flow
import oneflow.unittest


def _of_image_normalize(images, image_static_shape, std, mean):
    image_zeros = np.zeros(tuple(image_static_shape))
    for (idx, image) in enumerate(images):
        image_zeros[idx, : image.shape[1], : image.shape[2], : image.shape[3]] = image
    image_tensors = flow.tensor(
        image_zeros, dtype=flow.float, device=flow.device("cpu")
    )
    image_tensor_buffer = flow.tensor_to_tensor_buffer(image_tensors, instance_dims=3)
    image_normalizer = flow.nn.image.normalize(std, mean)
    norm_images = image_normalizer(image_tensor_buffer)
    return norm_images.numpy()


def _read_images_by_cv(image_files):
    images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
    return [np.expand_dims(image, axis=0) for image in images]


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0)
    assert isinstance(
        image_static_shape, np.ndarray
    ), "image_shapes: {}, image_static_shape: {}".format(
        str(image_shapes), str(image_static_shape)
    )
    image_static_shape = image_static_shape.tolist()
    assert image_static_shape[0] == 1, str(image_static_shape)
    image_static_shape[0] = len(image_shapes)
    return image_static_shape


def _compare_image_normalize(test_case, image_files, std, mean):
    images = _read_images_by_cv(image_files)
    assert all([len(image.shape) == 4 for image in images])
    image_static_shape = _get_images_static_shape(images)
    norm_images = _of_image_normalize(images, image_static_shape, std, mean)
    std_array = np.array(std).reshape(1, 1, 1, -1)
    mean_array = np.array(mean).reshape(1, 1, 1, -1)
    for (image, norm_image) in zip(images, norm_images):
        np_norm_image = np.squeeze((image - mean_array) / std_array, axis=0)
        norm_image = norm_image[
            : np_norm_image.shape[0], : np_norm_image.shape[1], : np_norm_image.shape[2]
        ]
        test_case.assertTrue(np.allclose(np_norm_image, norm_image))


@flow.unittest.skip_unless_1n1d()
class TestImageNormalize(flow.unittest.TestCase):
    def test_image_normalize(test_case):
        _compare_image_normalize(
            test_case,
            [
                "/dataset/mscoco_2017/val2017/000000000139.jpg",
                "/dataset/mscoco_2017/val2017/000000000632.jpg",
            ],
            (102.9801, 115.9465, 122.7717),
            (1.0, 1.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
