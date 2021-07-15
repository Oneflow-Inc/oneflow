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
import numpy as np
import oneflow as flow
from PIL import Image
import oneflow.typing as oft


def _of_image_decode(images):
    image_files = [open(im, "rb") for im in images]
    images_bytes = [imf.read() for imf in image_files]
    static_shape = (len(images_bytes), max([len(bys) for bys in images_bytes]))
    for imf in image_files:
        imf.close()

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def image_decode_job(
        images_def: oft.ListListNumpy.Placeholder(shape=static_shape, dtype=flow.int8)
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(images_def)
        decoded_images_buffer = flow.image_decode(images_buffer)
        return flow.tensor_buffer_to_tensor_list(
            decoded_images_buffer, shape=(640, 640, 3), dtype=flow.uint8
        )

    images_np_arr = [
        np.frombuffer(bys, dtype=np.byte).reshape(1, -1) for bys in images_bytes
    ]
    decoded_images = image_decode_job([images_np_arr]).get().numpy_lists()
    return decoded_images[0]


def _compare_jpg_decode_with_pil(test_case, images, print_debug_info=False):
    r"""
    The jpg image's decoded results with opencv and pil image are slightly different,
    their green channels have difference of 1.
    """
    of_decoded_images = _of_image_decode(images)
    pil_images = [Image.open(image) for image in images]
    # convert image to BGR
    pil_decoded_images = [np.array(image)[:, :, ::-1] for image in pil_images]

    for of_decoded_image, pil_decoded_image in zip(
        of_decoded_images, pil_decoded_images
    ):
        of_decoded_image = of_decoded_image.squeeze()
        test_case.assertTrue(len(of_decoded_image.shape) == 3)
        test_case.assertTrue(len(pil_decoded_image.shape) == 3)

        diff = of_decoded_image - pil_decoded_image
        diff_index = np.where(diff != 0)
        diff_abs_values = diff[diff_index]

        if print_debug_info:
            print("of_decoded_image:\n", of_decoded_image, of_decoded_image.shape)
            print("pil_decoded_image:\n", pil_decoded_image, pil_decoded_image.shape)
            print("diff_index:\n", diff_index)
            print("diff_abs_values:\n", diff_abs_values)
            print(
                "of_decoded_image diff:\n",
                of_decoded_image[diff_index[0], diff_index[1]],
            )
            print(
                "pil_decoded_image diff:\n",
                pil_decoded_image[diff_index[0], diff_index[1]],
            )

        # only green channel has difference of 1
        test_case.assertTrue(np.all(diff_index[-1] == 1))
        test_case.assertTrue(np.all(diff_abs_values == 1))


# @flow.unittest.skip_unless_1n1d()
# TODO(zhangwenxiao, jiangxuefei): refine in multi-client
@unittest.skipIf(True, "skip for now because of single-client tensor_list removed")
class TestImageDecode(flow.unittest.TestCase):
    def test_image_decode(test_case):
        _compare_jpg_decode_with_pil(
            test_case,
            [
                "/dataset/mscoco_2017/val2017/000000000139.jpg",
                "/dataset/mscoco_2017/val2017/000000000632.jpg",
            ],
            # True,
        )


if __name__ == "__main__":
    unittest.main()
