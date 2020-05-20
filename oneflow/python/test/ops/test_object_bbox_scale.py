import oneflow as flow
import numpy as np
import random
import os
import cv2


def _random_sample_images(anno_file, image_dir, batch_size):
    from pycocotools.coco import COCO

    image_files = []
    image_ids = []
    batch_group_id = -1

    coco = COCO(anno_file)
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
    images = [cv2.imread(image_file).astype(np.single) for image_file in image_files]
    bbox_list = _get_images_bbox_list(coco, image_ids)
    return images, bbox_list


def _get_images_bbox_list(coco, image_ids):
    bbox_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        anno_ids = list(filter(lambda anno_id: coco.anns[anno_id]["iscrowd"] == 0, anno_ids))
        bbox_array = np.array([coco.anns[anno_id]["bbox"] for anno_id in anno_ids], dtype=np.single)
        bbox_list.append(bbox_array)
    return bbox_list


def _get_images_static_shape(images):
    image_shapes = [image.shape for image in images]
    image_static_shape = np.amax(image_shapes, axis=0).tolist()
    return [len(image_shapes)] + image_static_shape


def _get_bbox_static_shape(bbox_list):
    bbox_shapes = [bbox.shape for bbox in bbox_list]
    bbox_static_shape = np.amax(bbox_shapes, axis=0)
    return [len(bbox_list)] + bbox_static_shape.tolist()


def _of_target_resize_bbox_scale(images, bbox_list, target_size, max_size):
    image_shape = _get_images_static_shape(images)
    bbox_shape = _get_bbox_static_shape(bbox_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def target_resize_bbox_scale_job(
        image_def=flow.MirroredTensorListDef(shape=tuple(image_shape), dtype=flow.float),
        bbox_def=flow.MirroredTensorListDef(shape=tuple(bbox_shape), dtype=flow.float),
    ):
        images_buffer = flow.tensor_list_to_tensor_buffer(image_def)
        resized_images_buffer, new_size, scale = flow.image_target_resize(
            images_buffer, target_size, max_size
        )
        bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
        scaled_bbox = flow.object_bbox_scale(bbox_buffer, scale)
        scaled_bbox_list = flow.tensor_buffer_to_tensor_list(
            scaled_bbox, shape=bbox_shape[1:], dtype=flow.float
        )
        return scaled_bbox_list, new_size

    input_image_list = [np.expand_dims(image, axis=0) for image in images]
    input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
    output_bbox_list, output_image_size = target_resize_bbox_scale_job(
        [input_image_list], [input_bbox_list]
    ).get()
    return output_bbox_list.ndarray_lists()[0], output_image_size.ndarray_list()[0]


def _compare_bbox_scale(
    test_case, anno_file, image_dir, batch_size, target_size, max_size, print_debug_info=False
):
    images, bbox_list = _random_sample_images(anno_file, image_dir, batch_size)
    of_bbox_list, image_size_list = _of_target_resize_bbox_scale(
        images, bbox_list, target_size, max_size
    )

    for image, bbox, of_bbox, image_size in zip(images, bbox_list, of_bbox_list, image_size_list):
        h, w = image_size
        oh, ow = image.shape[0:2]
        scale_h = h / oh
        scale_w = w / ow
        bbox[:, 0] *= scale_w
        bbox[:, 1] *= scale_h
        bbox[:, 2] *= scale_w
        bbox[:, 3] *= scale_h
        test_case.assertTrue(np.allclose(bbox, of_bbox))


def test_object_bbox_scale(test_case):
    _compare_bbox_scale(
        test_case,
        "/dataset/mscoco_2017/annotations/instances_val2017.json",
        "/dataset/mscoco_2017/val2017",
        4,
        800,
        1333,
    )
