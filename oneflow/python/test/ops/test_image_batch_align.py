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
import oneflow.typing as oft


def _of_image_batch_align(images, input_shape, output_shape, alignment):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_batch_align_job(
        images_def: oft.ListListNumpy.Placeholder(shape=input_shape, dtype=flow.float)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        image = flow.image_batch_align(
            images_buffer, shape=output_shape[1:], dtype=flow.float, alignment=alignment
        )
        return image

    image = image_batch_align_job([images]).get()
    return image.numpy_list()[0]


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


def _roundup(x, n):
    return int((x + n - 1) / n) * n


def _compare_image_batch_align(
    test_case, image_files, alignment, print_debug_info=False
):
    images = _read_images_by_cv(image_files)
    image_shape = _get_images_static_shape(images)
    assert len(image_shape) == 4
    aligned_image_shape = [
        image_shape[0],
        _roundup(image_shape[1], alignment),
        _roundup(image_shape[2], alignment),
        image_shape[3],
    ]

    if print_debug_info:
        print("image_shape:", image_shape)
        print("aligned_image_shape:", aligned_image_shape)

    image_tensor = _of_image_batch_align(
        images, tuple(image_shape), tuple(aligned_image_shape), alignment
    )
    test_case.assertTrue(np.array_equal(aligned_image_shape, image_tensor.shape))

    empty_image_array = np.zeros(aligned_image_shape, np.single)
    for empty_image, image in zip(empty_image_array, images):
        image = image.squeeze()
        empty_image[0 : image.shape[0], 0 : image.shape[1], :] = image

    test_case.assertTrue(np.array_equal(image_tensor, empty_image_array))


@flow.unittest.skip_unless_1n1d()
class TestImageBatchAlign(flow.unittest.TestCase):
    def test_image_batch_align(test_case):
        _compare_image_batch_align(
            test_case,
            [
                "/dataset/mscoco_2017/val2017/000000000139.jpg",
                "/dataset/mscoco_2017/val2017/000000000632.jpg",
                "/dataset/mscoco_2017/val2017/000000000785.jpg",
                "/dataset/mscoco_2017/val2017/000000001000.jpg",
            ],
            16,
            # True,
        )


if __name__ == "__main__":
    unittest.main()
