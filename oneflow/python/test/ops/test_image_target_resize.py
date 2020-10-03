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
from PIL import Image
import oneflow.typing as oft


def _of_image_target_resize(images, image_static_shape, target_size, max_size):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_target_resize_job(
        images_def: oft.ListListNumpy.Placeholder(
            shape=image_static_shape, dtype=flow.float
        )
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        resized_images_buffer, size, scale = flow.image_target_resize(
            images_buffer,
            target_size=target_size,
            max_size=max_size,
            resize_side="shorter",
        )
        resized_images = flow.tensor_buffer_to_tensor_list(
            resized_images_buffer,
            shape=(target_size, max_size, image_static_shape[-1]),
            dtype=flow.float,
        )
        return resized_images, size, scale

    resized_images, size, scale = image_target_resize_job([images]).get()
    resized_images = resized_images.numpy_lists()[0]
    size = size.numpy_list()[0]
    scale = scale.numpy_list()[0]
    return resized_images, size, scale


def _read_images_by_pil(image_files):
    images = [Image.open(image_file) for image_file in image_files]
    # convert image to BGR
    converted_images = [
        np.array(image).astype(np.single)[:, :, ::-1] for image in images
    ]
    return [np.expand_dims(image, axis=0) for image in converted_images]


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


def _target_resize_by_cv(images, target_size, max_size):
    resized_images = []
    resized_sizes = []
    resized_scales = []
    for image in images:
        squeeze_image = image.squeeze()
        resized_size, resized_scale = _get_target_resize_size(
            squeeze_image.shape[1], squeeze_image.shape[0], target_size, max_size
        )
        resized_images.append(cv2.resize(squeeze_image, resized_size))
        resized_sizes.append(resized_size)
        resized_scales.append(resized_scale)

    return resized_images, resized_sizes, resized_scales


def _get_target_resize_size(w, h, target_size, max_size):
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))

    min_resized_size = target_size
    max_resized_size = int(
        round(max_original_size / min_original_size * min_resized_size)
    )
    if max_size > 0 and max_resized_size > max_size:
        max_resized_size = max_size
        min_resized_size = int(
            round(max_resized_size * min_original_size / max_original_size)
        )

    if w < h:
        res_w = min_resized_size
        res_h = max_resized_size
    else:
        res_w = max_resized_size
        res_h = min_resized_size

    scale_w = res_w / w
    scale_h = res_h / h
    return (res_w, res_h), (scale_w, scale_h)


def _compare_image_target_resize_with_cv(
    test_case, image_files, target_size, max_size, print_debug_info=False
):
    images = _read_images_by_cv(image_files)
    image_static_shape = _get_images_static_shape(images)

    resized_images, size, scale = _of_image_target_resize(
        images, tuple(image_static_shape), target_size, max_size
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


@unittest.skip("TODO(tsai): ask wx for help")
@flow.unittest.skip_unless_1n1d()
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
