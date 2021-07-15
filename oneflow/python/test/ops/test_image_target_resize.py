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
import typing as tp
import oneflow as flow
import oneflow.typing as otp
import image_test_util


def _of_image_target_resize(
    images, target_size, max_size, image_static_shape, aspect_ratio_list
):
    assert image_static_shape[-1] == 3

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_target_resize_job(
        image: otp.ListListNumpy.Placeholder(shape=image_static_shape, dtype=flow.float)
    ) -> tp.Tuple[otp.ListListNumpy, otp.ListNumpy, otp.ListNumpy]:
        image_buffer = flow.tensor_list_to_tensor_buffer(image)
        res_image_buffer, new_size, scale = flow.image_target_resize(
            image_buffer,
            target_size=target_size,
            max_size=max_size,
            resize_side="shorter",
        )
        out_shape = image_test_util.infer_keep_aspect_ratio_resized_images_static_shape(
            target_size=target_size,
            min_size=None,
            max_size=max_size,
            aspect_ratio_list=aspect_ratio_list,
            resize_side="shorter",
            channels=3,
        )
        res_image = flow.tensor_buffer_to_tensor_list(
            res_image_buffer, shape=out_shape, dtype=flow.float,
        )
        return res_image, new_size, scale

    res_image, new_size, scale = image_target_resize_job([images])
    return res_image[0], new_size[0], scale[0]


def _target_resize_by_cv(images, target_size, max_size):
    res_images = []
    res_sizes = []
    res_scales = []
    for image in images:
        h, w = image.shape[0:2]
        res_size, res_scale = _get_target_resize_size(w, h, target_size, max_size)
        res_images.append(cv2.resize(image, res_size))
        res_sizes.append(res_size)
        res_scales.append(res_scale)

    return res_images, res_sizes, res_scales


def _get_target_resize_size(w, h, target_size, max_size):
    aspect_ratio = float(min((w, h))) / float(max((w, h)))
    (
        min_res_size,
        max_res_size,
    ) = image_test_util.compute_keep_aspect_ratio_resized_size(
        target_size, None, max_size, aspect_ratio, "shorter"
    )

    if w < h:
        res_w = min_res_size
        res_h = max_res_size
    else:
        res_w = max_res_size
        res_h = min_res_size

    scale_w = res_w / w
    scale_h = res_h / h
    return (res_w, res_h), (scale_w, scale_h)


def _compare_image_target_resize_with_cv(
    test_case, image_files, target_size, max_size, print_debug_info=False
):
    images = image_test_util.read_images_by_cv(image_files, flow.float)
    image_static_shape, aspect_ratio_list = image_test_util.infer_images_static_shape(
        images
    )
    expand_images = [np.expand_dims(image, axis=0) for image in images]

    resized_images, size, scale = _of_image_target_resize(
        expand_images, target_size, max_size, image_static_shape, aspect_ratio_list
    )

    cv_resized_images, cv_resized_sizes, cv_resized_scales = _target_resize_by_cv(
        images, target_size, max_size
    )

    for (
        resized_image,
        cv_resized_image,
        image_size,
        image_scale,
        resized_size,
        resized_scale,
    ) in zip(
        resized_images,
        cv_resized_images,
        size,
        scale,
        cv_resized_sizes,
        cv_resized_scales,
    ):
        if print_debug_info:
            print("resized_image shape:", resized_image.shape)
            print("cv_resized_image shape:", cv_resized_image.shape)
            print("resized w & h:", image_size, resized_size)
            print("resize w_scale & h_scale:", image_scale, resized_scale)

        test_case.assertTrue(np.allclose(resized_image, cv_resized_image))
        test_case.assertTrue(np.allclose(image_size, resized_size))
        test_case.assertTrue(np.allclose(image_scale, resized_scale))


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
class TestImageTargetResize(flow.unittest.TestCase):
    def test_image_target_resize(test_case):
        _compare_image_target_resize_with_cv(
            test_case,
            [
                "/dataset/mscoco_2017/val2017/000000000139.jpg",
                "/dataset/mscoco_2017/val2017/000000000632.jpg",
            ],
            800,
            1333,
            # True,
        )


if __name__ == "__main__":
    unittest.main()
