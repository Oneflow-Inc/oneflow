import oneflow as flow
import numpy as np
import random


def _of_object_bbox_flip(bbox_list, image_size, flip_code):
    bbox_shape = _get_bbox_static_shape(bbox_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.function(func_config)
    def object_bbox_flip_job(
        bbox_def=flow.MirroredTensorListDef(shape=tuple(bbox_shape), dtype=flow.float),
        image_size_def=flow.MirroredTensorDef(shape=image_size.shape, dtype=flow.int32),
    ):
        bbox_buffer = flow.tensor_list_to_tensor_buffer(bbox_def)
        flip_bbox = flow.object_bbox_flip(bbox_buffer, image_size_def, flip_code)
        return flow.tensor_buffer_to_tensor_list(flip_bbox, shape=bbox_shape[1:], dtype=flow.float)

    input_bbox_list = [np.expand_dims(bbox, axis=0) for bbox in bbox_list]
    bbox_tensor = object_bbox_flip_job([input_bbox_list], [image_size]).get()
    return bbox_tensor.ndarray_lists()[0]


def _get_bbox_static_shape(bbox_list):
    bbox_shapes = [bbox.shape for bbox in bbox_list]
    bbox_static_shape = np.amax(bbox_shapes, axis=0)
    return [len(bbox_list)] + bbox_static_shape.tolist()


def _compare_bbox_flip(test_case, anno_file, batch_size, flip_code, print_debug_info=False):
    from pycocotools.coco import COCO

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()
    rand_img_ids = random.sample(img_ids, batch_size)

    bbox_list = []
    image_size_list = []
    for img_id in rand_img_ids:
        image_size_list.append([coco.imgs[img_id]["height"], coco.imgs[img_id]["width"]])
        anno_ids = list(
            filter(
                lambda anno_id: coco.anns[anno_id]["iscrowd"] == 0, coco.getAnnIds(imgIds=[img_id])
            )
        )
        bbox_array = np.array([coco.anns[anno_id]["bbox"] for anno_id in anno_ids], dtype=np.single)
        bbox_list.append(bbox_array)

    image_size_array = np.array(image_size_list, dtype=np.int32)
    of_bbox_list = _of_object_bbox_flip(bbox_list, image_size_array, flip_code)
    for of_bbox, bbox, image_size in zip(of_bbox_list, bbox_list, image_size_list):
        h, w = image_size
        if flip_code == 1:
            xmin = bbox[:, 0].copy()
            xmax = bbox[:, 2].copy()
            bbox[:, 0] = w - xmax - 1
            bbox[:, 2] = w - xmin - 1
        else:
            raise NotImplementedError

        if print_debug_info:
            print("-" * 20)
            print("ret_bbox:\n", of_bbox.squeeze())
            print("bbox:\n", bbox)

        test_case.assertTrue(np.allclose(of_bbox.squeeze(), bbox))


def test_object_bbox_flip(test_case):
    _compare_bbox_flip(test_case, "/dataset/mscoco_2017/annotations/instances_val2017.json", 4, 1)
