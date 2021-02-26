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
import os
import random

import cv2
import numpy as np
import oneflow as flow
import oneflow.typing as oft

coco_dict = dict()


def _coco(anno_file):
    global coco_dict

    if anno_file not in coco_dict:
        from pycocotools.coco import COCO

        coco_dict[anno_file] = COCO(anno_file)

    return coco_dict[anno_file]


def _random_sample_image_ids(coco, batch_size):
    img_ids = coco.getImgIds()

    batch_img_ids = []
    batch_group_id = -1
    while len(batch_img_ids) < batch_size:
        rand_img_id = random.choice(img_ids)
        img_h = coco.imgs[rand_img_id]["height"]
        img_w = coco.imgs[rand_img_id]["width"]
        group_id = int(img_h / img_w)

        if batch_group_id == -1:
            batch_group_id = group_id

        if group_id != batch_group_id:
            continue

        anno_ids = coco.getAnnIds(imgIds=[rand_img_id])
        if len(anno_ids) == 0:
            continue

        batch_img_ids.append(rand_img_id)

    return batch_img_ids


def _read_images_with_cv(coco, image_dir, image_ids):
    image_files = [
        os.path.join(image_dir, coco.imgs[img_id]["file_name"]) for img_id in image_ids
    ]
    return [cv2.imread(image_file).astype(np.single) for image_file in image_files]


def _get_images_segm_poly(coco, image_ids):
    img_segm_poly_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        assert len(anno_ids) > 0, "img {} has no anno".format(img_id)

        segm_poly_list = []
        for anno_id in anno_ids:
            anno = coco.anns[anno_id]
            if anno["iscrowd"] != 0:
                continue
            segm = anno["segmentation"]
            assert isinstance(segm, list)
            assert len(segm) > 0, str(len(segm))
            assert all([len(poly) > 0 for poly in segm]), str(
                [len(poly) for poly in segm]
            )
            segm_poly_list.append(segm)

        img_segm_poly_list.append(segm_poly_list)

    return img_segm_poly_list


def _get_check_image_size(coco, image_ids, images):
    assert len(image_ids) == len(images)
    image_size_list = []
    for i, img_id in enumerate(image_ids):
        img_h = coco.imgs[img_id]["height"]
        img_w = coco.imgs[img_id]["width"]
        assert img_h == images[i].shape[0]
        assert img_w == images[i].shape[1]
        image_size_list.append([img_w, img_h])

    return image_size_list


def _segm_poly_to_tensor(img_segm_poly_list):
    poly_array_list = []
    poly_index_array_list = []
    for img_idx, segm_poly_list in enumerate(img_segm_poly_list):
        img_poly_elem_list = []
        img_poly_index_list = []

        for obj_idx, poly_list in enumerate(segm_poly_list):
            for poly_idx, poly in enumerate(poly_list):
                img_poly_elem_list.extend(poly)
                for pt_idx, pt in enumerate(poly):
                    if pt_idx % 2 == 0:
                        img_poly_index_list.append([pt_idx / 2, poly_idx, obj_idx])

        img_poly_array = np.array(img_poly_elem_list, dtype=np.single).reshape(-1, 2)
        assert img_poly_array.size > 0, segm_poly_list
        poly_array_list.append(img_poly_array)

        img_poly_index_array = np.array(img_poly_index_list, dtype=np.int32)
        assert img_poly_index_array.size > 0, segm_poly_list
        poly_index_array_list.append(img_poly_index_array)

    return poly_array_list, poly_index_array_list


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0)
    assert isinstance(
        image_static_shape, np.ndarray
    ), "image_shapes: {}, image_static_shape: {}".format(
        str(image_shapes), str(image_static_shape)
    )
    image_static_shape = image_static_shape.tolist()
    image_static_shape.insert(0, len(image_shapes))
    return image_static_shape


def _get_segm_poly_static_shape(poly_list, poly_index_list):
    assert len(poly_list) == len(poly_index_list)
    num_images = len(poly_list)
    max_poly_elems = 0
    for poly, poly_index in zip(poly_list, poly_index_list):
        assert len(poly.shape) == 2
        assert len(poly_index.shape) == 2, str(poly_index.shape)
        assert poly.shape[0] == poly_index.shape[0]
        assert poly.shape[1] == 2
        assert poly_index.shape[1] == 3
        max_poly_elems = max(max_poly_elems, poly.shape[0])
    return [num_images, max_poly_elems, 2], [num_images, max_poly_elems, 3]


def _of_poly_to_mask_pipline(
    images, poly_list, poly_index_list, num_segms_list, target_size, max_size
):
    assert len(images) == len(poly_list)
    assert len(poly_list) == len(poly_index_list)
    image_shape = _get_images_static_shape(images)
    poly_shape, poly_index_shape = _get_segm_poly_static_shape(
        poly_list, poly_index_list
    )
    max_num_segms = max(num_segms_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.mirrored_view())
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def poly_to_mask_job(
        image_def: oft.ListListNumpy.Placeholder(
            shape=tuple(image_shape), dtype=flow.float
        ),
        poly_def: oft.ListListNumpy.Placeholder(
            shape=tuple(poly_shape), dtype=flow.float
        ),
        poly_index_def: oft.ListListNumpy.Placeholder(
            shape=tuple(poly_index_shape), dtype=flow.int32
        ),
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
        resized_images_buffer, new_size, scale = flow.image_target_resize(
            images_buffer, target_size=target_size, max_size=max_size
        )
        poly_buffer = flow.tensor_list_to_tensor_buffer(poly_def)
        poly_index_buffer = flow.tensor_list_to_tensor_buffer(poly_index_def)
        scaled_poly_buffer = flow.object_segmentation_polygon_scale(poly_buffer, scale)
        mask_buffer = flow.object_segmentation_polygon_to_mask(
            scaled_poly_buffer, poly_index_buffer, new_size
        )
        mask_list = flow.tensor_buffer_to_tensor_list(
            mask_buffer, shape=(max_num_segms, target_size, max_size), dtype=flow.int8
        )
        scaled_poly_list = flow.tensor_buffer_to_tensor_list(
            scaled_poly_buffer, shape=poly_shape[1:], dtype=flow.float
        )
        return mask_list, scaled_poly_list

    input_image_list = [np.expand_dims(image, axis=0) for image in images]
    input_poly_list = [np.expand_dims(poly, axis=0) for poly in poly_list]
    input_poly_index_list = [
        np.expand_dims(poly_index, axis=0) for poly_index in poly_index_list
    ]
    output_mask_list, output_poly_list = poly_to_mask_job(
        [input_image_list], [input_poly_list], [input_poly_index_list]
    ).get()

    return output_mask_list.numpy_lists()[0], output_poly_list.numpy_lists()[0]


def _get_target_resize_scale(size, target_size, max_size):
    w, h = size
    min_ori_size = float(min((w, h)))
    max_ori_size = float(max((w, h)))

    min_res_size = target_size
    max_res_size = int(round(max_ori_size / min_ori_size * min_res_size))
    if max_res_size > max_size:
        max_res_size = max_size
        min_res_size = int(round(max_res_size * min_ori_size / max_ori_size))

    if w < h:
        res_w = min_res_size
        res_h = max_res_size
    else:
        res_w = max_res_size
        res_h = min_res_size

    return [res_w, res_h], [res_w / w, res_h / h]


def _scale_poly_list(img_segm_poly_list, scale_list):
    assert len(img_segm_poly_list) == len(scale_list)
    for img_idx, segm_poly_list in enumerate(img_segm_poly_list):
        scale_w, scale_h = scale_list[img_idx]
        for poly_list in segm_poly_list:
            for poly in poly_list:
                for pt_idx in range(len(poly)):
                    if pt_idx % 2 == 0:
                        poly[pt_idx] *= scale_w
                    else:
                        poly[pt_idx] *= scale_h

    return img_segm_poly_list


def _poly_to_mask_with_cv(img_segm_poly_list, image_size_list):
    assert len(img_segm_poly_list) == len(image_size_list)

    img_segm_mask_list = []
    for segm_poly_list, size in zip(img_segm_poly_list, image_size_list):
        segm_mask_list = []
        for poly_list in segm_poly_list:
            segm_mask = np.zeros(shape=size[::-1], dtype=np.int8)
            poly_array_list = [
                np.int32(np.round(np.asarray(poly)).reshape(-1, 2))
                for poly in poly_list
            ]
            cv2.fillPoly(segm_mask, poly_array_list, 1, lineType=8)
            segm_mask_list.append(segm_mask)

        segm_mask_array = np.asarray(segm_mask_list)
        img_segm_mask_list.append(segm_mask_array)

    return img_segm_mask_list


def _poly_to_mask_with_of_and_cv(
    test_case,
    anno_file,
    image_dir,
    batch_size,
    target_size,
    max_size,
    img_ids=None,
    print_debug_info=False,
):
    coco = _coco(anno_file)
    if img_ids is not None:
        assert len(img_ids) == batch_size
        rand_img_ids = img_ids
    else:
        rand_img_ids = _random_sample_image_ids(coco, batch_size)
    images = _read_images_with_cv(coco, image_dir, rand_img_ids)
    image_size_list = _get_check_image_size(coco, rand_img_ids, images)
    img_segm_poly_list = _get_images_segm_poly(coco, rand_img_ids)
    assert len(img_segm_poly_list) == len(image_size_list)

    poly_list, poly_index_list = _segm_poly_to_tensor(img_segm_poly_list)
    num_segms_list = [len(segm_poly_list) for segm_poly_list in img_segm_poly_list]
    if print_debug_info:
        print("poly_shapes:", [poly.shape for poly in poly_list])
        print("poly_index_shapes", [poly_index.shape for poly_index in poly_index_list])
        print("image_size_list:", image_size_list)
        print("num_segms_list:", num_segms_list)

    of_mask_list, of_scaled_poly_list = _of_poly_to_mask_pipline(
        images, poly_list, poly_index_list, num_segms_list, target_size, max_size
    )
    of_mask_list = [
        mask_array.reshape(-1, mask_array.shape[-2], mask_array.shape[-1])
        for mask_array in of_mask_list
    ]

    if print_debug_info:
        print("of_mask_list shapes:", [of_mask.shape for of_mask in of_mask_list])

    # manual test
    new_image_size_list = []
    scale_list = []
    for image_size in image_size_list:
        new_size, scale = _get_target_resize_scale(image_size, target_size, max_size)
        new_image_size_list.append(new_size)
        scale_list.append(scale)

    if print_debug_info:
        print("resized size: {}, scale: {}".format(new_image_size_list, scale_list))

    scaled_img_segm_poly_list = _scale_poly_list(img_segm_poly_list, scale_list)
    scaled_poly_list, scaled_poly_index_list = _segm_poly_to_tensor(
        scaled_img_segm_poly_list
    )
    img_segm_mask_list = _poly_to_mask_with_cv(
        scaled_img_segm_poly_list, new_image_size_list
    )
    assert len(img_segm_mask_list) == len(of_mask_list)

    if test_case is not None:
        for of_scaled_poly, scaled_poly, poly_index, scaled_poly_index in zip(
            of_scaled_poly_list,
            scaled_poly_list,
            poly_index_list,
            scaled_poly_index_list,
        ):
            if print_debug_info:
                print(
                    "compare scaled poly: shape {} vs {}\n\tmax_abs_diff: {}".format(
                        of_scaled_poly.shape,
                        scaled_poly.shape,
                        np.max(np.absolute(of_scaled_poly - scaled_poly)),
                    )
                )

            test_case.assertTrue(np.allclose(of_scaled_poly, scaled_poly))
            test_case.assertTrue(np.array_equal(poly_index, scaled_poly_index))

        for of_mask, gt_mask in zip(of_mask_list, img_segm_mask_list):
            if print_debug_info:
                print(
                    "compare segm mask: shape {} vs {}".format(
                        of_mask.shape, gt_mask.shape
                    )
                )

            test_case.assertTrue(np.array_equal(of_mask.shape, gt_mask.shape))

    return of_mask_list, img_segm_mask_list


def _vis_img_segm_mask_cmp(mask_list, cmp_mask_list):
    assert len(mask_list) == len(cmp_mask_list)

    import matplotlib.pyplot as plt
    import ipywidgets as ipw

    from IPython.display import display, clear_output

    plt.close("all")
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_dpi(150)

    out_widget = ipw.Output()
    next_btn = ipw.Button(description="Next")
    vbox = ipw.VBox(children=(out_widget, next_btn))
    display(vbox)

    cur_img_idx = 0
    cur_mask_idx = 0

    def display_fig():
        nonlocal cur_img_idx, cur_mask_idx
        mask_array = mask_list[cur_img_idx]
        cmp_mask_array = cmp_mask_list[cur_img_idx]
        assert mask_array.shape == cmp_mask_array.shape, "{} vs {}".format(
            str(mask_array.shape), str(cmp_mask_array.shape)
        )
        mask = mask_array[cur_mask_idx]
        cmp_mask = cmp_mask_array[cur_mask_idx]

        ax1.clear()
        ax2.clear()
        fig.suptitle(
            "img_idx:{}, mask_idx:{}".format(cur_img_idx, cur_mask_idx), fontsize=10
        )
        ax1.imshow(mask)
        ax2.imshow(cmp_mask)

        nonlocal out_widget
        with out_widget:
            clear_output(wait=True)
            display(fig)

    def on_next_clicked(b):
        nonlocal cur_img_idx, cur_mask_idx
        eof = False
        cur_mask_array_len = len(mask_list[cur_img_idx])
        if cur_mask_idx < cur_mask_array_len - 1:
            cur_mask_idx += 1
        else:
            cur_mask_list_len = len(mask_list)
            if cur_img_idx < cur_mask_list_len - 1:
                cur_img_idx += 1
                cur_mask_idx = 0
            else:
                eof = True

        if eof:
            nonlocal next_btn
            next_btn.close()
            del next_btn
        else:
            display_fig()

    next_btn.on_click(on_next_clicked)
    display_fig()


def _check_empty_anno_img_ids(anno_file):
    coco = _coco(anno_file)
    img_ids = coco.getImgIds()
    empty_anno_img_ids = []
    for img_id in img_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        if len(anno_ids) == 0:
            empty_anno_img_ids.append(img_id)

    print("empty_anno_img_ids:", empty_anno_img_ids)


if __name__ == "__main__":
    # _check_empty_anno_img_ids("/dataset/mscoco_2017/annotations/instances_val2017.json")
    of_mask_list, mask_list = _poly_to_mask_with_of_and_cv(
        None,
        "/dataset/mscoco_2017/annotations/instances_val2017.json",
        "/dataset/mscoco_2017/val2017",
        4,
        800,
        1333,
        # img_ids=[226111, 58636, 458790, 461275],
        print_debug_info=True,
    )
    _vis_img_segm_mask_cmp(of_mask_list, mask_list)


@flow.unittest.skip_unless_1n1d()
class TestObjectSegmPolyToMask(flow.unittest.TestCase):
    def test_poly_to_mask(test_case):
        _poly_to_mask_with_of_and_cv(
            test_case,
            "/dataset/mscoco_2017/annotations/instances_val2017.json",
            "/dataset/mscoco_2017/val2017",
            4,
            800,
            1333,
            # print_debug_info=True,
        )


if __name__ == "__main__":
    unittest.main()
