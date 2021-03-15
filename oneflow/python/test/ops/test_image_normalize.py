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


def _of_image_normalize(images, image_shape, std, mean):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_normalize_job(
        images_def: oft.ListListNumpy.Placeholder(shape=image_shape, dtype=flow.float)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        norm_images = flow.image_normalize(images_buffer, std, mean)
        return flow.tensor_buffer_to_tensor_list(
            norm_images, shape=image_shape[1:], dtype=flow.float
        )

    image_tensor = image_normalize_job([images]).get()
    return image_tensor.numpy_lists()[0]


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
    image_shape = _get_images_static_shape(images)

    norm_images = _of_image_normalize(images, tuple(image_shape), std, mean)

    std_array = np.array(std).reshape(1, 1, 1, -1)
    mean_array = np.array(mean).reshape(1, 1, 1, -1)

    for image, norm_image in zip(images, norm_images):
        exp_norm_image = (image - mean_array) / std_array
        test_case.assertTrue(np.allclose(exp_norm_image, norm_image))


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
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
