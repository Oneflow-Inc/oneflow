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
import random
import os
import typing as tp
import oneflow as flow
import oneflow.typing as otp

global_coco_dict = dict()
default_coco_anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
default_coco_image_dir = "/dataset/mscoco_2017/val2017"


def _make_image_resize_to_fixed_func(
    target_size,
    image_static_shape,
    dtype,
    origin_dtype=flow.float32,
    channels=3,
    interpolation_type="bilinear",
    func_cfg=None,
    print_debug_info=False,
):
    @flow.global_function(type="predict", function_config=func_cfg)
    def image_resize_to_fixed(
        image_list: otp.ListListNumpy.Placeholder(
            shape=image_static_shape, dtype=origin_dtype
        )
    ) -> tp.Tuple[otp.ListNumpy, otp.ListNumpy]:
        image_buffer = flow.tensor_list_to_tensor_buffer(image_list)
        res_image, scale, _ = flow.image.resize(
            image_buffer,
            target_size=target_size,
            keep_aspect_ratio=False,
            channels=channels,
            dtype=dtype,
            interpolation_type=interpolation_type,
        )

        return res_image, scale

    return image_resize_to_fixed


def _make_image_resize_keep_aspect_ratio_func(
    target_size,
    min_size,
    max_size,
    image_static_shape,
    origin_image_size_range,
    dtype,
    channels=3,
    resize_side="shorter",
    interpolation_type="bilinear",
    func_cfg=None,
    print_debug_info=False,
):
    @flow.global_function(type="predict", function_config=func_cfg)
    def image_resize_keep_aspect_ratio(
        image_list: otp.ListListNumpy.Placeholder(
            shape=image_static_shape, dtype=dtype
        ),
    ) -> tp.Tuple[otp.ListListNumpy, otp.ListNumpy, otp.ListNumpy]:
        image_buffer = flow.tensor_list_to_tensor_buffer(image_list)
        res_image, scale, new_size = flow.image.resize(
            image_buffer,
            target_size=target_size,
            min_size=min_size,
            max_size=max_size,
            keep_aspect_ratio=True,
            resize_side=resize_side,
            interpolation_type=interpolation_type,
        )

        out_shape = _infer_resized_image_static_shape(
            target_size, origin_image_size_range, resize_side, channels,
        )
        if print_debug_info:
            print("resized image_static_shape: {}".format(out_shape))

        res_image = flow.tensor_buffer_to_tensor_list(
            res_image, shape=out_shape, dtype=dtype,
        )

        return res_image, scale, new_size

    return image_resize_keep_aspect_ratio


def _of_image_resize(
    image_list,
    dtype=flow.float32,
    origin_dtype=None,
    channels=3,
    keep_aspect_ratio=False,
    target_size=None,
    min_size=None,
    max_size=None,
    resize_side="shorter",
    interpolation_type="bilinear",
    print_debug_info=False,
):
    assert isinstance(image_list, (list, tuple))
    assert all(isinstance(image, np.ndarray) for image in image_list)
    assert all(image.ndim == 3 for image in image_list)
    assert all(image.shape[2] == channels for image in image_list)

    image_static_shape, image_size_range = _infer_images_static_shape(
        image_list, channels
    )
    if print_debug_info:
        print("image_static_shape: {}".format(image_static_shape))
        print("image_size_range: {}".format(image_size_range))

    flow.clear_default_session()
    func_cfg = flow.FunctionConfig()
    func_cfg.default_logical_view(flow.scope.mirrored_view())

    image_list = [np.expand_dims(image, axis=0) for image in image_list]
    if keep_aspect_ratio:
        image_resize_func = _make_image_resize_keep_aspect_ratio_func(
            target_size=target_size,
            min_size=min_size,
            max_size=max_size,
            image_static_shape=image_static_shape,
            origin_image_size_range=image_size_range,
            dtype=dtype,
            channels=channels,
            resize_side=resize_side,
            interpolation_type=interpolation_type,
            func_cfg=func_cfg,
            print_debug_info=print_debug_info,
        )
        res_image, scale, new_size = image_resize_func([image_list])
        return (res_image[0], scale[0], new_size[0])
    else:
        if origin_dtype is None:
            origin_dtype = dtype

        image_resize_func = _make_image_resize_to_fixed_func(
            target_size=target_size,
            image_static_shape=image_static_shape,
            dtype=dtype,
            origin_dtype=origin_dtype,
            channels=channels,
            interpolation_type=interpolation_type,
            func_cfg=func_cfg,
            print_debug_info=print_debug_info,
        )
        res_image, scale = image_resize_func([image_list])
        new_size = np.asarray([(target_size, target_size)] * len(image_list))
        return (res_image[0], scale[0], new_size)


def _infer_images_static_shape(images, channels=3):
    image_shapes = [image.shape for image in images]
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == channels for image in images)
    image_shapes = np.asarray(image_shapes)
    max_h = np.max(image_shapes[:, 0]).item()
    max_w = np.max(image_shapes[:, 1]).item()
    image_static_shape = (len(images), max_h, max_w, channels)

    group_ids = [int(image.shape[0] / image.shape[1]) for image in images]
    min_h = np.min(image_shapes[:, 0]).item()
    min_w = np.min(image_shapes[:, 1]).item()
    assert all(group_id == group_ids[0] for group_id in group_ids)
    if group_ids[0] == 0:
        image_size_range = (min_h, max_h, min_w, max_w)
    else:
        image_size_range = (min_w, max_w, min_h, max_h)

    return image_static_shape, image_size_range


def _infer_resized_image_static_shape(
    target_size,
    origin_size_range=None,
    resize_side="shorter",
    channels=3,
    keep_aspect_ratio=True,
):
    assert isinstance(origin_size_range, (list, tuple))
    assert len(origin_size_range) == 4

    min_aspect_ratio = origin_size_range[0] / origin_size_range[3]
    max_aspect_ratio = origin_size_range[1] / origin_size_range[2]

    res_shape = None
    if keep_aspect_ratio:
        assert isinstance(target_size, int)
        if resize_side == "shorter":
            max_size = int(round(target_size / min_aspect_ratio))
            res_shape = (target_size, max_size, channels)
        elif resize_side == "longer":
            min_size = int(round(target_size * max_aspect_ratio))
            res_shape = (min_size, target_size, channels)
        else:
            raise NotImplementedError
    else:
        assert isinstance(target_size, (list, tuple))
        assert len(target_size) == 2
        assert all(isinstance(size, int) for size in target_size)
        res_w, res_h = target_size
        res_shape = (res_h, res_w, channels)

    return res_shape


def _cv_read_images_from_files(image_files, dtype):
    np_dtype = flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
    images = [cv2.imread(image_file).astype(np_dtype) for image_file in image_files]
    assert all(isinstance(image, np.ndarray) for image in images)
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == 3 for image in images)
    return images


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

        min_org_size = float(min((w, h)))
        max_org_size = float(max((w, h)))
        aspect_ratio = min_org_size / max_org_size

        if resize_side == "shorter":
            min_res_size = target_size
            max_res_size = int(round(min_res_size / aspect_ratio))
            if max_size is not None and max_res_size > max_size:
                max_res_size = max_size
                min_res_size = int(round(max_res_size * aspect_ratio))

        elif resize_side == "longer":
            max_res_size = target_size
            min_res_size = int(round(max_res_size * aspect_ratio))
            if min_size is not None and min_res_size < min_size:
                min_res_size = min_size
                max_res_size = int(round(min_res_size / aspect_ratio))

        else:
            raise NotImplementedError

        if w < h:
            res_w = min_res_size
            res_h = max_res_size
        else:
            res_w = max_res_size
            res_h = min_res_size

    else:
        assert isinstance(target_size, (list, tuple))
        assert len(target_size) == 2
        assert all(isinstance(size, int) for size in target_size)
        res_w, res_h = target_size

    scale_w = res_w / w
    scale_h = res_h / h
    return (res_w, res_h), (scale_w, scale_h)


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
        h, w = image.shape[:2]
        new_size, scale = _get_resize_size_and_scale(
            w, h, target_size, min_size, max_size, keep_aspect_ratio, resize_side
        )
        res_image_list.append(
            cv2.resize(image.squeeze(), new_size, interpolation=interpolation).astype(
                dtype
            )
        )
        res_size_list.append(new_size)
        res_scale_list.append(scale)

    return res_image_list, res_scale_list, res_size_list


def _coco(anno_file):
    global global_coco_dict

    if anno_file not in global_coco_dict:
        from pycocotools.coco import COCO

        global_coco_dict[anno_file] = COCO(anno_file)

    return global_coco_dict[anno_file]


def _coco_random_sample_images(
    anno_file=default_coco_anno_file, image_dir=default_coco_image_dir, batch_size=2
):
    image_files = []
    image_ids = []
    batch_group_id = -1

    coco = _coco(anno_file)
    img_ids = coco.getImgIds()

    while len(image_files) < batch_size:
        rand_img_id = random.choice(img_ids)
        img_h = coco.imgs[rand_img_id]["height"]
        img_w = coco.imgs[rand_img_id]["width"]
        group_id = int(img_h / img_w)

        if batch_group_id == -1:
            batch_group_id = group_id

        if group_id != batch_group_id:
            continue

        image_files.append(os.path.join(image_dir, coco.imgs[rand_img_id]["file_name"]))
        image_ids.append(rand_img_id)

    assert len(image_files) == len(image_ids)
    return image_files, image_ids


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
    print_debug_info=False,
):
    if origin_dtype is None:
        origin_dtype = dtype
    image_list = _cv_read_images_from_files(image_files, origin_dtype)

    if print_debug_info:
        print("origin images shapes: {}".format([image.shape for image in image_list]))
        print(
            "target_size: {}, min_size: {}, max_size: {}, keep_aspect_ratio: {}, \n"
            "resize_side: {}, dtype: {}, origin_dtype: {}".format(
                target_size,
                min_size,
                max_size,
                keep_aspect_ratio,
                resize_side,
                dtype,
                origin_dtype,
            )
        )

    of_res_images, of_scales, of_new_sizes = _of_image_resize(
        image_list=image_list,
        dtype=dtype,
        origin_dtype=origin_dtype,
        keep_aspect_ratio=keep_aspect_ratio,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        resize_side=resize_side,
        print_debug_info=print_debug_info,
    )

    cv_res_images, cv_scales, cv_new_sizes = _cv_image_resize(
        image_list=image_list,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
        min_size=min_size,
        max_size=max_size,
        resize_side=resize_side,
        dtype=flow.convert_oneflow_dtype_to_numpy_dtype(dtype),
    )

    if print_debug_info:
        print("comparing resized image between of and cv")
        for i, (of_image, cv_image) in enumerate(zip(of_res_images, cv_res_images)):
            print("    origin image shape: {}".format(image_list[i].shape))
            print(
                "    resized image shape: {} vs. {}".format(
                    of_image.shape, cv_image.shape
                )
            )
            # print("    of_res_image:\n{}".format(of_res_image))
            # print("    cv_res_image:\n{}".format(cv_res_image))

        print("comparing resized image scale between of and cv")
        for of_scale, cv_scale in zip(of_scales, cv_scales):
            print("    scale: {} vs. {}:".format(of_scale, cv_scale))

        print("comparing resized image new size between of and cv")
        for of_new_size, cv_new_size in zip(of_new_sizes, cv_new_sizes):
            print("    new_size: {} vs. {}:".format(of_new_size, cv_new_size))

    for (
        of_res_image,
        cv_res_image,
        of_scale,
        cv_scale,
        of_new_size,
        cv_new_size,
    ) in zip(
        of_res_images, cv_res_images, of_scales, cv_scales, of_new_sizes, cv_new_sizes,
    ):
        test_case.assertTrue(np.allclose(of_res_image, cv_res_image))
        test_case.assertTrue(np.allclose(of_scale, cv_scale))
        test_case.assertTrue(np.allclose(of_new_size, cv_new_size))


@unittest.skip("TODO(tsai): ask wx for help")
@flow.unittest.skip_unless_1n1d()
class TestImageResize(flow.unittest.TestCase):
    def test_image_resize_to_fixed_size(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=(224, 224),
            keep_aspect_ratio=False,
            # print_debug_info=True,
        )

    def test_image_resize_shorter_to_target_size(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=800,
            keep_aspect_ratio=True,
            resize_side="shorter",
            # print_debug_info=True,
        )

    def test_image_resize_longer_to_target_size(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            keep_aspect_ratio=True,
            resize_side="longer",
            # print_debug_info=True,
        )

    def test_image_resize_shorter_to_target_size_with_max_size(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=800,
            max_size=1333,
            keep_aspect_ratio=True,
            resize_side="shorter",
            # print_debug_info=True,
        )

    def test_image_resize_longer_to_target_size_with_min_size(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            min_size=600,
            keep_aspect_ratio=True,
            resize_side="longer",
            # print_debug_info=True,
        )

    def test_image_resize_to_fixed_size_with_dtype_uint8(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=(1000, 1000),
            keep_aspect_ratio=False,
            dtype=flow.uint8,
            # print_debug_info=True,
        )

    def test_image_resize_shorter_to_target_size_with_max_size_with_dtype_uint8(
        test_case,
    ):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=1000,
            max_size=1600,
            keep_aspect_ratio=True,
            resize_side="shorter",
            dtype=flow.uint8,
            # print_debug_info=True,
        )

    def test_image_resize_uint8_to_float(test_case):
        image_files, _ = _coco_random_sample_images()
        _test_image_resize_with_cv(
            test_case,
            image_files,
            target_size=(1000, 1000),
            keep_aspect_ratio=False,
            dtype=flow.float32,
            origin_dtype=flow.uint8,
            # print_debug_info=True,
        )


if __name__ == "__main__":
    unittest.main()
