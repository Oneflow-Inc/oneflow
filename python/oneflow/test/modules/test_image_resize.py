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
import image_test_util
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


def _of_image_resize(
    image_list,
    dtype=flow.float32,
    origin_dtype=flow.float32,
    channels=3,
    keep_aspect_ratio=False,
    target_size=None,
    min_size=None,
    max_size=None,
    resize_side="shorter",
    interpolation_type="bilinear",
):
    assert isinstance(image_list, (list, tuple))
    assert all((isinstance(image, np.ndarray) for image in image_list))
    assert all((image.ndim == 3 for image in image_list))
    assert all((image.shape[2] == channels for image in image_list))
    res_image_list = []
    res_size_list = []
    res_scale_list = []
    image_resize_module = nn.image.Resize(
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        keep_aspect_ratio=keep_aspect_ratio,
        resize_side=resize_side,
        dtype=dtype,
        interpolation_type=interpolation_type,
        channels=channels,
    )
    for image in image_list:
        tensor_dtype = dtype if keep_aspect_ratio else origin_dtype
        input = flow.tensor(
            np.expand_dims(image, axis=0), dtype=tensor_dtype, device=flow.device("cpu")
        )
        image_buffer = flow.tensor_to_tensor_buffer(input, instance_dims=3)
        (res_image, scale, new_size) = image_resize_module(image_buffer)
        res_image = res_image.numpy()
        scale = scale.numpy()
        if not keep_aspect_ratio:
            new_size = np.asarray([(target_size, target_size)])
        else:
            new_size = new_size.numpy()
        res_image_list.append(res_image[0])
        res_size_list.append(new_size[0])
        res_scale_list.append(scale[0])
    return (res_image_list, res_scale_list, res_size_list)


def _get_resize_size_and_scale(
    w,
    h,
    target_size,
    min_size=None,
    max_size=None,
    keep_aspect_ratio=True,
    resize_side="shorter",
):
    if keep_aspect_ratio:
        assert isinstance(target_size, int)
        aspect_ratio = float(min((w, h))) / float(max((w, h)))
        (
            min_res_size,
            max_res_size,
        ) = image_test_util.compute_keep_aspect_ratio_resized_size(
            target_size, min_size, max_size, aspect_ratio, resize_side
        )
        if w < h:
            res_w = min_res_size
            res_h = max_res_size
        else:
            res_w = max_res_size
            res_h = min_res_size
    else:
        assert isinstance(target_size, (list, tuple))
        assert len(target_size) == 2
        assert all((isinstance(size, int) for size in target_size))
        (res_w, res_h) = target_size
    scale_w = res_w / w
    scale_h = res_h / h
    return ((res_w, res_h), (scale_w, scale_h))


def _cv_image_resize(
    image_list,
    target_size,
    keep_aspect_ratio=True,
    min_size=None,
    max_size=None,
    resize_side="shorter",
    interpolation=cv2.INTER_LINEAR,
    dtype=np.float32,
):
    res_image_list = []
    res_size_list = []
    res_scale_list = []
    for image in image_list:
        (h, w) = image.shape[:2]
        (new_size, scale) = _get_resize_size_and_scale(
            w, h, target_size, min_size, max_size, keep_aspect_ratio, resize_side
        )
        res_image_list.append(
            cv2.resize(image.squeeze(), new_size, interpolation=interpolation).astype(
                dtype
            )
        )
        res_size_list.append(new_size)
        res_scale_list.append(scale)
    return (res_image_list, res_scale_list, res_size_list)


def _test_image_resize_with_cv(
    test_case,
    image_files,
    target_size,
    min_size=None,
    max_size=None,
    keep_aspect_ratio=True,
    resize_side="shorter",
    dtype=flow.float32,
    origin_dtype=None,
):
    if origin_dtype is None:
        origin_dtype = dtype
    image_list = image_test_util.read_images_by_cv(image_files, origin_dtype)
    (of_res_images, of_scales, of_new_sizes) = _of_image_resize(
        image_list=image_list,
        dtype=dtype,
        origin_dtype=origin_dtype,
        keep_aspect_ratio=keep_aspect_ratio,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        resize_side=resize_side,
    )
    (cv_res_images, cv_scales, cv_new_sizes) = _cv_image_resize(
        image_list=image_list,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
        min_size=min_size,
        max_size=max_size,
        resize_side=resize_side,
        dtype=flow.convert_oneflow_dtype_to_numpy_dtype(dtype),
    )
    for (
        of_res_image,
        cv_res_image,
        of_scale,
        cv_scale,
        of_new_size,
        cv_new_size,
    ) in zip(
        of_res_images, cv_res_images, of_scales, cv_scales, of_new_sizes, cv_new_sizes
    ):
        test_case.assertTrue(np.allclose(of_res_image, cv_res_image))
        test_case.assertTrue(np.allclose(of_scale, cv_scale))
        test_case.assertTrue(np.allclose(of_new_size, cv_new_size))


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestImageResize(flow.unittest.TestCase):
    def test_image_resize_to_fixed_size(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case, image_files, target_size=(224, 224), keep_aspect_ratio=False
        )

    def test_image_resize_shorter_to_target_size(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=800,
            keep_aspect_ratio=True,
            resize_side="shorter",
        )

    def test_image_resize_longer_to_target_size(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            keep_aspect_ratio=True,
            resize_side="longer",
        )

    def test_image_resize_shorter_to_target_size_with_max_size(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=800,
            max_size=1333,
            keep_aspect_ratio=True,
            resize_side="shorter",
        )

    def test_image_resize_longer_to_target_size_with_min_size(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            min_size=600,
            keep_aspect_ratio=True,
            resize_side="longer",
        )

    def test_image_resize_to_fixed_size_with_dtype_uint8(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=(1000, 1000),
            keep_aspect_ratio=False,
            dtype=flow.uint8,
        )

    def test_image_reisze_shorter_to_target_size_with_max_size_with_dtype_uint8(
        test_case,
    ):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            max_size=1600,
            keep_aspect_ratio=True,
            resize_side="shorter",
            dtype=flow.uint8,
        )

    def test_image_resize_uint8_to_float(test_case):
        (image_files, _) = image_test_util.random_sample_images_from_coco()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=(1000, 1000),
            keep_aspect_ratio=False,
            dtype=flow.float32,
            origin_dtype=flow.uint8,
        )


if __name__ == "__main__":
    unittest.main()
