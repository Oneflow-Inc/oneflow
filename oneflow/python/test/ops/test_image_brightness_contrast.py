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
import cv2
import numpy as np
import oneflow as flow
import oneflow.typing as oft


def _of_image_brightness_contrast(
    images, image_shape, brightness, contrast, contrast_center
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_brightness_contrast_job(
        images_def: oft.ListListNumpy.Placeholder(shape=image_shape, dtype=flow.float)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        aug_images = flow.image.brightness_contrast(
            images_buffer,
            brightness=brightness,
            contrast=contrast,
            contrast_center=contrast_center,
            dtype=flow.uint8,
        )
        return flow.tensor_buffer_to_tensor_list(
            aug_images, shape=image_shape[1:], dtype=flow.float
        )

    image_tensor = image_brightness_contrast_job([images]).get()
    return image_tensor.numpy_lists()[0]


def _read_images_by_cv(image_files):

    images = [cv2.imread(image_file).astype(np.float32) for image_file in image_files]
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


def _compare_image_brightness_contrast(
    test_case, image_files, brightness, contrast, contrast_center
):
    images = _read_images_by_cv(image_files)
    assert all([len(image.shape) == 4 for image in images])
    image_shape = _get_images_static_shape(images)

    aug_images = _of_image_brightness_contrast(
        images, tuple(image_shape), brightness, contrast, contrast_center
    )

    np.set_printoptions(precision=6, suppress=True)
    for image, aug_image in zip(images, aug_images):
        multiplier = brightness * contrast
        addend = brightness * (contrast_center - contrast * contrast_center)
        # hack for different behavior between std::round and numpy.around
        exp_aug_image = image * multiplier + addend + 1e-4
        exp_aug_image = np.clip(np.around(exp_aug_image), 0.0, 255.0)
        test_case.assertTrue(
            np.allclose(exp_aug_image, aug_image, rtol=1e-03, atol=1e-04)
        )


def test_image_brightness_contrast_1(test_case):
    _compare_image_brightness_contrast(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        brightness=1.5,
        contrast=3.0,
        contrast_center=128.0,
    )


def test_image_brightness_contrast_2(test_case):
    _compare_image_brightness_contrast(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        brightness=1.0,
        contrast=1.0,
        contrast_center=128.0,
    )


def test_image_brightness_contrast_3(test_case):
    _compare_image_brightness_contrast(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        brightness=0.9,
        contrast=0.7,
        contrast_center=128.0,
    )
