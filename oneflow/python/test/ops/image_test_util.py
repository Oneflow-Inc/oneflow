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
import numpy as np
import cv2
import oneflow as flow
import PIL
import random
import os

global_coco_dict = dict()
default_coco_anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
default_coco_image_dir = "/dataset/mscoco_2017/val2017"


def get_coco(anno_file):
    global global_coco_dict

    if anno_file not in global_coco_dict:
        from pycocotools.coco import COCO

        global_coco_dict[anno_file] = COCO(anno_file)

    return global_coco_dict[anno_file]


def random_sample_images_from_coco(
    anno_file=default_coco_anno_file, image_dir=default_coco_image_dir, batch_size=2
):
    image_files = []
    image_ids = []
    batch_group_id = -1

    coco = get_coco(anno_file)
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


def read_images_by_cv(image_files, dtype, channels=3):
    np_dtype = flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
    images = [cv2.imread(image_file).astype(np_dtype) for image_file in image_files]
    assert all(isinstance(image, np.ndarray) for image in images)
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == channels for image in images)
    return images


def read_images_by_pil(image_files, dtype, channels=3):
    image_objs = [PIL.Image.open(image_file) for image_file in image_files]
    images = []
    np_dtype = flow.convert_oneflow_dtype_to_numpy_dtype(dtype)

    for im in image_objs:
        bands = im.getbands()
        band = "".join(bands)
        if band == "RGB":
            # convert to BGR
            images.append(np.asarray(im).astype(np_dtype)[:, :, ::-1])
        elif band == "L":
            gs_image = np.asarray(im).astype(np_dtype)
            gs_image_shape = gs_image.shape
            assert len(gs_image_shape) == 2
            gs_image = gs_image.reshape(gs_image_shape + (1,))
            gs_image = np.broadcast_to(gs_image, shape=gs_image_shape + (3,))
            images.append(gs_image)
        elif band == "BGR":
            images.append(np.asarray(im).astype(np_dtype))
        else:
            raise NotImplementedError

    assert all(isinstance(image, np.ndarray) for image in images)
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == channels for image in images)

    return images


def infer_images_static_shape(images, channels=3):
    image_shapes = [image.shape for image in images]
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == channels for image in images)
    image_shapes = np.asarray(image_shapes)

    max_h = np.max(image_shapes[:, 0]).item()
    max_w = np.max(image_shapes[:, 1]).item()
    image_static_shape = (len(images), max_h, max_w, channels)

    group_ids = []  # 0: h < w, 1: h >= w
    aspect_ratio_list = []  # shorter / longer
    for image_shape in image_shapes:
        h, w = image_shape[0:2]
        if h < w:
            group_id = 0
            aspect_ratio = h / w
        else:
            group_id = 1
            aspect_ratio = w / h
        group_ids.append(group_id)
        aspect_ratio_list.append(aspect_ratio)
    assert all(group_id == group_ids[0] for group_id in group_ids)

    return image_static_shape, aspect_ratio_list


def compute_keep_aspect_ratio_resized_size(
    target_size, min_size, max_size, aspect_ratio, resize_side
):
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

    return (min_res_size, max_res_size)


def infer_keep_aspect_ratio_resized_images_static_shape(
    target_size,
    min_size,
    max_size,
    aspect_ratio_list,
    resize_side="shorter",
    channels=3,
):
    resized_size_list = []
    for aspect_ratio in aspect_ratio_list:
        resized_size_list.append(
            compute_keep_aspect_ratio_resized_size(
                target_size, min_size, max_size, aspect_ratio, resize_side
            )
        )

    res_min_size, res_max_size = max(
        resized_size_list, key=lambda size: size[0] * size[1]
    )
    return (res_min_size, res_max_size, channels)
