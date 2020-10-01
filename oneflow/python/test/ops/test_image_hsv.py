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


def _of_image_hsv(images, image_shape, hue, saturation, value):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_hsv_job(
        images_def: oft.ListListNumpy.Placeholder(shape=image_shape, dtype=flow.float)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        aug_images = flow.image.hsv(
            images_buffer,
            hue=hue,
            saturation=saturation,
            value=value,
            dtype=flow.uint8,
        )
        return flow.tensor_buffer_to_tensor_list(
            aug_images, shape=image_shape[1:], dtype=flow.float
        )

    image_tensor = image_hsv_job([images]).get()
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


def _compare_image_hsv(test_case, image_files, hue, saturation, value):
    images = _read_images_by_cv(image_files)
    assert all([len(image.shape) == 4 for image in images])
    image_shape = _get_images_static_shape(images)

    aug_images = _of_image_hsv(images, tuple(image_shape), hue, saturation, value)

    h_rad = hue * np.pi / 180
    hue_mat = np.array(
        [
            [1, 0, 0],
            [0, np.cos(h_rad), np.sin(h_rad)],
            [0, -np.sin(h_rad), np.cos(h_rad)],
        ],
        dtype=np.float32,
    )
    sat_mat = np.array(
        [[1, 0, 0], [0, saturation, 0], [0, 0, saturation]], dtype=np.float32
    )
    val_mat = np.array([[value, 0, 0], [0, value, 0], [0, 0, value]], dtype=np.float32)
    rgb2yiq_np = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]],
        dtype=np.float32,
    )
    yiq2rgb_np = np.array(
        [[1, 0.956, 0.621], [1, -0.272, -0.647], [1, -1.107, 1.705]], dtype=np.float32
    )
    transform_mat = yiq2rgb_np @ hue_mat @ sat_mat @ val_mat @ rgb2yiq_np

    for image, aug_image in zip(images, aug_images):
        exp_aug_image = np.tensordot(image, transform_mat, axes=([3], [1]))
        exp_aug_image = np.clip(np.around(exp_aug_image), 0.0, 255.0)
        test_case.assertTrue(np.allclose(exp_aug_image, aug_image, atol=1))


def test_image_hsv_1(test_case):
    _compare_image_hsv(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        hue=0.0,
        saturation=1.0,
        value=1.0,
    )


def test_image_hsv_2(test_case):
    _compare_image_hsv(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        hue=0.2,
        saturation=1.3,
        value=1.0,
    )


def test_image_hsv_3(test_case):
    _compare_image_hsv(
        test_case,
        [
            "/dataset/mscoco_2017/val2017/000000000139.jpg",
            "/dataset/mscoco_2017/val2017/000000000632.jpg",
        ],
        hue=-0.3,
        saturation=0.8,
        value=1.0,
    )
