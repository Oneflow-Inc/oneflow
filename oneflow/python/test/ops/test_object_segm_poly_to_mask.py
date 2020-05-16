import oneflow as flow
import numpy as np
import random
import os
import cv2

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

        batch_img_ids.append(rand_img_id)

    return batch_img_ids


def _read_images_with_cv(coco, image_dir, image_ids):
    image_files = [os.path.join(image_dir, coco.imgs[img_id]["file_name"]) for img_id in image_ids]
    return [cv2.imread(image_file).astype(np.single) for image_file in image_files]


def _get_images_segm_poly(coco, image_ids):
    img_segm_poly_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        anno_ids = list(filter(lambda anno_id: coco.anns[anno_id]["iscrowd"] == 0, anno_ids))
        segm_poly_list = []
        for anno_id in anno_ids:
            segm = coco.anns[anno_id]["segmentation"]
            assert isinstance(segm, list)
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
        image_size_list.append([img_h, img_w])
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
        poly_array_list.append(img_poly_array)
        img_poly_index_array = np.array(img_poly_index_list, dtype=np.int32)
        poly_index_array_list.append(img_poly_index_array)

    return poly_array_list, poly_index_array_list


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0).tolist()
    return [len(image_shapes)] + image_static_shape


def _get_segm_poly_static_shape(poly_list, poly_index_list):
    assert len(poly_list) == len(poly_index_list)
    num_images = len(poly_list)
    max_poly_elems = 0
    for poly, poly_index in zip(poly_list, poly_index_list):
        assert len(poly.shape) == 2
        assert len(poly_index.shape) == 2
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
    poly_shape, poly_index_shape = _get_segm_poly_static_shape(poly_list, poly_index_list)
    max_num_segms = max(num_segms_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def poly_to_mask_job(
        image_def=flow.MirroredTensorListDef(shape=tuple(image_shape), dtype=flow.float),
        poly_def=flow.MirroredTensorListDef(shape=tuple(poly_shape), dtype=flow.float),
        poly_index_def=flow.MirroredTensorListDef(shape=tuple(poly_index_shape), dtype=flow.int32),
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
        resized_images_buffer, new_size, scale = flow.image_target_resize(
            images_buffer, target_size, max_size
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
    input_poly_index_list = [np.expand_dims(poly_index, axis=0) for poly_index in poly_index_list]
    output_mask_list, output_poly_list = poly_to_mask_job(
        [input_image_list], [input_poly_list], [input_poly_index_list]
    ).get()

    return output_mask_list.ndarray_lists()[0], output_poly_list.ndarray_lists()[0]


def _get_target_resize_scale(size, target_size, max_size):
    h, w = size
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

    return [res_h, res_w], [res_h / h, res_w / w]


def _scale_poly_list(img_segm_poly_list, scale_list):
    assert len(img_segm_poly_list) == len(scale_list)
    for img_idx, segm_poly_list in enumerate(img_segm_poly_list):
        scale_h, scale_w = scale_list[img_idx]
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
            segm_mask = np.zeros(size, dtype=np.int8)
            poly_array_list = [
                np.int32(np.round(np.asarray(poly)).reshape(-1, 2)) for poly in poly_list
            ]
            cv2.fillPoly(segm_mask, poly_array_list, 1, lineType=8)
            segm_mask_list.append(segm_mask)
        segm_mask_array = np.asarray(segm_mask_list)
        img_segm_mask_list.append(segm_mask_array)

    return img_segm_mask_list


def _poly_to_mask_with_of_and_cv(
    test_case, anno_file, image_dir, batch_size, target_size, max_size, print_debug_info=False
):
    coco = _coco(anno_file)
    rand_img_ids = _random_sample_image_ids(coco, batch_size)
    images = _read_images_with_cv(coco, image_dir, rand_img_ids)
    image_size_list = _get_check_image_size(coco, rand_img_ids, images)
    img_segm_poly_list = _get_images_segm_poly(coco, rand_img_ids)
    assert len(img_segm_poly_list) == len(image_size_list)

    poly_list, poly_index_list = _segm_poly_to_tensor(img_segm_poly_list)
    num_segms_list = [len(segm_poly_list) for segm_poly_list in img_segm_poly_list]
    of_mask_list, of_scaled_poly_list = _of_poly_to_mask_pipline(
        images, poly_list, poly_index_list, num_segms_list, target_size, max_size
    )

    if print_debug_info:
        print("poly_shapes:", [poly.shape for poly in poly_list])
        print("poly_index_shapes", [poly_index.shape for poly_index in poly_index_list])
        print("image_size_list:\n", image_size_list)
        print("num_segms_list:\n", num_segms_list)
        print("of_mask_list shapes:", [of_mask.shape for of_mask in of_mask_list])

    # manual test
    new_image_size_list = []
    scale_list = []
    for image_size in image_size_list:
        new_size, scale = _get_target_resize_scale(image_size, target_size, max_size)
        new_image_size_list.append(new_size)
        scale_list.append(scale)
    scaled_img_segm_poly_list = _scale_poly_list(img_segm_poly_list, scale_list)
    scaled_poly_list, scaled_poly_index_list = _segm_poly_to_tensor(scaled_img_segm_poly_list)

    for of_scaled_poly, scaled_poly, poly_index, scaled_poly_index in zip(
        of_scaled_poly_list, scaled_poly_list, poly_index_list, scaled_poly_index_list
    ):
        test_case.assertTrue(np.allclose(of_scaled_poly, scaled_poly))
        test_case.assertTrue(np.array_equal(poly_index, scaled_poly_index))

    img_segm_mask_list = _poly_to_mask_with_cv(scaled_img_segm_poly_list, new_image_size_list)
    assert len(img_segm_mask_list) == len(of_mask_list)
    for of_mask, gt_mask in zip(of_mask_list, img_segm_mask_list):
        test_case.assertTrue(np.array_equal(of_mask.squeeze().shape, gt_mask.shape))

    return of_mask_list, img_segm_mask_list


def _vis_img_segm_mask_cmp(mask_list, of_mask_list):
    import matplotlib.pyplot as plt

    assert len(mask_list) == len(of_mask_list)
    frame = plt.gcf()
    rows = len(mask_list)
    cols = 2

    for img_idx, (mask, of_mask) in enumerate(zip(mask_list, of_mask_list)):
        frame.add_subplot(rows, cols, img_idx * 2 + 1)
        plt.imshow(mask)
        frame.add_subplot(rows, cols, img_idx * 2 + 2)
        plt.imshow(of_mask)

    plt.show()


def test_poly_to_mask(test_case):
    _poly_to_mask_with_of_and_cv(
        test_case,
        "/dataset/mscoco_2017/annotations/instances_val2017.json",
        "/dataset/mscoco_2017/val2017",
        4,
        800,
        1333,
        # True,
    )
