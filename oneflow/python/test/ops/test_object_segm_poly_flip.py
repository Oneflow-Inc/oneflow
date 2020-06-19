import random

import numpy as np
import oneflow as flow


def _of_object_segm_poly_flip(poly_list, image_size, flip_code):
    poly_shape = _get_segm_poly_static_shape(poly_list)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def object_segm_poly_flip_job(
        poly_def=flow.MirroredTensorListDef(shape=tuple(poly_shape), dtype=flow.float),
        image_size_def=flow.MirroredTensorDef(shape=image_size.shape, dtype=flow.int32),
    ):
        poly_buffer = flow.tensor_list_to_tensor_buffer(poly_def)
        flip_poly = flow.object_segmentation_polygon_flip(
            poly_buffer, image_size_def, flip_code
        )
        return flow.tensor_buffer_to_tensor_list(
            flip_poly, shape=poly_shape[1:], dtype=flow.float
        )

    input_poly_list = [np.expand_dims(bbox, axis=0) for bbox in poly_list]
    poly_tensor = object_segm_poly_flip_job([input_poly_list], [image_size]).get()
    return poly_tensor.ndarray_lists()[0]


def _get_segm_poly_static_shape(poly_list):
    poly_shapes = [poly.shape for poly in poly_list]
    poly_static_shape = np.amax(poly_shapes, axis=0)
    assert isinstance(
        poly_static_shape, np.ndarray
    ), "poly_shapes: {}, poly_static_shape: {}".format(
        str(poly_shapes), str(poly_static_shape)
    )
    poly_static_shape = poly_static_shape.tolist()
    poly_static_shape.insert(0, len(poly_list))
    return poly_static_shape


def _compare_segm_poly_flip(
    test_case, anno_file, batch_size, flip_code, print_debug_info=False
):
    from pycocotools.coco import COCO

    coco = COCO(anno_file)
    img_ids = coco.getImgIds()

    segm_poly_list = []
    image_size_list = []
    sample_cnt = 0
    while sample_cnt < batch_size:
        rand_img_id = random.choice(img_ids)
        anno_ids = coco.getAnnIds(imgIds=[rand_img_id])
        if len(anno_ids) == 0:
            continue

        poly_pts = []
        for anno_id in anno_ids:
            anno = coco.anns[anno_id]
            if anno["iscrowd"] != 0:
                continue
            assert isinstance(anno["segmentation"], list)
            for poly in anno["segmentation"]:
                assert isinstance(poly, list)
                poly_pts.extend(poly)

        poly_array = np.array(poly_pts, dtype=np.single).reshape(-1, 2)
        segm_poly_list.append(poly_array)
        image_size_list.append(
            [coco.imgs[rand_img_id]["height"], coco.imgs[rand_img_id]["width"]]
        )
        sample_cnt += 1

    image_size_array = np.array(image_size_list, dtype=np.int32)
    of_segm_poly_list = _of_object_segm_poly_flip(
        segm_poly_list, image_size_array, flip_code
    )
    for of_poly, poly, image_size in zip(
        of_segm_poly_list, segm_poly_list, image_size_list
    ):
        h, w = image_size
        if flip_code == 1:
            poly[:, 0] = w - poly[:, 0]
        else:
            raise NotImplementedError

        if print_debug_info:
            print("-" * 20)
            print("of_poly:", of_poly.squeeze().shape, "\n", of_poly.squeeze())
            print("poly:", poly.shape, "\n", poly)

    test_case.assertTrue(np.allclose(of_poly.squeeze(), poly))


def test_object_segm_poly_flip(test_case):
    _compare_segm_poly_flip(
        test_case, "/dataset/mscoco_2017/annotations/instances_val2017.json", 4, 1
    )
