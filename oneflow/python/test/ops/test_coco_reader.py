import numpy as np
import oneflow as flow
import os
import cv2

coco_dict = dict()


def _coco(anno_file):
    global coco_dict

    if anno_file not in coco_dict:
        from pycocotools.coco import COCO

        coco_dict[anno_file] = COCO(anno_file)

    return coco_dict[anno_file]


def _make_coco_reader_op(
    annotation_file,
    image_dir,
    batch_size,
    stride_partition=True,
    shuffle=True,
    random_seed=-1,
    name=None,
):
    if name is None:
        name = "COCOReader"
    assert isinstance(name, str)

    return (
        flow.user_op_builder(name)
        .Op("COCOReader")
        .Output("image")
        .Output("image_id")
        .Output("image_size")
        .Output("gt_bbox")
        .Output("gt_label")
        .Output("gt_segm")
        .Output("gt_segm_index")
        .Attr("annotation_file", annotation_file, "AttrTypeString")
        .Attr("image_dir", image_dir, "AttrTypeString")
        .Attr("batch_size", batch_size, "AttrTypeInt64")
        .Attr("shuffle_after_epoch", shuffle, "AttrTypeBool")
        .Attr("random_seed", random_seed, "AttrTypeInt64")
        .Attr("group_by_ratio", True, "AttrTypeBool")
        .Attr("stride_partition", stride_partition, "AttrTypeBool")
        .Attr("empty_tensor_size", 32, "AttrTypeInt64")
        .Attr("tensor_init_bytes", 1048576, "AttrTypeInt32")
        .Build()
    )


def _of_coco_data_load(anno_file, image_dir, nthread, batch_size, stride_partition):
    flow.config.cpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def coco_load_fn():
        with flow.fixed_placement("cpu", "0:0-{}".format(nthread - 1)):
            coco_reader = _make_coco_reader_op(
                annotation_file=anno_file,
                image_dir=image_dir,
                batch_size=batch_size,
                stride_partition=stride_partition,
                shuffle=False,
            )

            (
                image,
                image_id,
                image_size,
                gt_bbox,
                gt_label,
                gt_segm,
                gt_segm_index,
            ) = coco_reader.InferAndTryRun().RemoteBlobList()

            decoded_image = flow.image_decode(image)
            image_list = flow.tensor_buffer_to_tensor_list(
                decoded_image, shape=(640, 800, 3), dtype=flow.uint8
            )
            bbox_list = flow.tensor_buffer_to_tensor_list(gt_bbox, shape=(128, 4), dtype=flow.float)
            label_list = flow.tensor_buffer_to_tensor_list(gt_label, shape=(128,), dtype=flow.int32)
            segm_list = flow.tensor_buffer_to_tensor_list(gt_segm, shape=(512, 2), dtype=flow.float)
            segm_index_list = flow.tensor_buffer_to_tensor_list(
                gt_segm_index, shape=(512, 3), dtype=flow.int32
            )

        return image_id, image_size, image_list, bbox_list, label_list, segm_list, segm_index_list

    image_id, image_size, image, gt_bbox, gt_label, gt_segm, gt_segm_index = coco_load_fn().get()
    return (
        image_id.ndarray(),
        image_size.ndarray(),
        image.ndarray_lists(),
        gt_bbox.ndarray_lists(),
        gt_label.ndarray_lists(),
        gt_segm.ndarray_lists(),
        gt_segm_index.ndarray_lists(),
    )


def _get_coco_image_samples(anno_file, image_dir, image_ids):
    coco = _coco(anno_file)
    category_id_to_contiguous_id_map = _get_category_id_to_contiguous_id_map(coco)
    image, image_size = _read_images_with_cv(coco, image_dir, image_ids)
    bbox = _read_bbox(coco, image_ids)
    label = _read_label(coco, image_ids, category_id_to_contiguous_id_map)
    img_segm_poly_list = _read_segm_poly(coco, image_ids)
    poly, poly_index = _segm_poly_list_to_tensor(img_segm_poly_list)
    samples = []
    for im, ims, b, l, p, pi in zip(image, image_size, bbox, label, poly, poly_index):
        samples.append(dict(image=im, image_size=ims, bbox=b, label=l, poly=p, poly_index=pi))
    return samples


def _get_category_id_to_contiguous_id_map(coco):
    return {v: i + 1 for i, v in enumerate(coco.getCatIds())}


def _read_images_with_cv(coco, image_dir, image_ids):
    image_files = [os.path.join(image_dir, coco.imgs[img_id]["file_name"]) for img_id in image_ids]
    image_size = [(coco.imgs[img_id]["height"], coco.imgs[img_id]["width"]) for img_id in image_ids]
    return [cv2.imread(image_file).astype(np.single) for image_file in image_files], image_size


def _bbox_convert_from_xywh_to_xyxy(bbox, image_h, image_w):
    x, y, w, h = bbox
    x1, y1 = x, y
    x2 = x1 + max(w - 1, 0)
    y2 = y1 + max(h - 1, 0)

    # clip to image
    x1 = min(max(x1, 0), image_w - 1)
    y1 = min(max(y1, 0), image_h - 1)
    x2 = min(max(x2, 0), image_w - 1)
    y2 = min(max(y2, 0), image_h - 1)

    if x1 >= x2 or y1 >= y2:
        return None

    return [x1, y1, x2, y2]


def _read_bbox(coco, image_ids):
    img_bbox_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        assert len(anno_ids) > 0, "image with id {} has no anno".format(img_id)
        image_h = coco.imgs[img_id]["height"]
        image_w = coco.imgs[img_id]["width"]

        bbox_list = []
        for anno_id in anno_ids:
            anno = coco.anns[anno_id]
            if anno["iscrowd"] != 0:
                continue

            bbox = anno["bbox"]
            assert isinstance(bbox, list)
            bbox_ = _bbox_convert_from_xywh_to_xyxy(bbox, image_h, image_w)
            if bbox_ is not None:
                bbox_list.append(bbox_)

        bbox_array = np.array(bbox_list, dtype=np.single)
        img_bbox_list.append(bbox_array)

    return img_bbox_list


def _read_label(coco, image_ids, category_id_to_contiguous_id_map):
    img_label_list = []
    for img_id in image_ids:
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        assert len(anno_ids) > 0, "image with id {} has no anno".format(img_id)

        label_list = []
        for anno_id in anno_ids:
            anno = coco.anns[anno_id]
            if anno["iscrowd"] != 0:
                continue
            cate_id = anno["category_id"]
            isinstance(cate_id, int)
            label_list.append(category_id_to_contiguous_id_map[cate_id])
        label_array = np.array(label_list, dtype=np.int32)
        img_label_list.append(label_array)
    return img_label_list


def _read_segm_poly(coco, image_ids):
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
            assert all([len(poly) > 0 for poly in segm]), str([len(poly) for poly in segm])
            segm_poly_list.append(segm)

        img_segm_poly_list.append(segm_poly_list)

    return img_segm_poly_list


def _segm_poly_list_to_tensor(img_segm_poly_list):
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


def _coco_sorted_img_ids(anno_file):
    coco = _coco(anno_file)
    img_ids = coco.getImgIds()
    img_ids.sort()
    print("Info of the first 20 images:")
    for i, img_id in enumerate(img_ids[:20]):
        img_h = coco.imgs[img_id]["height"]
        img_w = coco.imgs[img_id]["width"]
        group_id = int(img_h / img_w)
        anno_ids = coco.getAnnIds(imgIds=[img_id])
        print(
            "index: {}, image_id: {}, group_id: {}, anno len: {}".format(
                i, img_id, group_id, len(anno_ids)
            )
        )

    return img_ids


def test_coco_reader(test_case):
    anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
    image_dir = "/dataset/mscoco_2017/val2017"
    # _coco_sorted_img_ids(anno_file)

    image_id, image_size, image, bbox, label, poly, poly_index = _of_coco_data_load(
        anno_file, image_dir, 1, 2, False
    )

    samples = _get_coco_image_samples(anno_file, image_dir, image_id)
    for i, sample in enumerate(samples):
        test_case.assertTrue(np.array_equal(image[0][i].squeeze(), sample["image"]))
        test_case.assertTrue(np.array_equal(image_size[i], sample["image_size"]))
        test_case.assertTrue(np.allclose(bbox[0][i].squeeze(), sample["bbox"]))
        test_case.assertTrue(np.array_equal(label[0][i].squeeze(), sample["label"]))
        test_case.assertTrue(np.allclose(poly[0][i].squeeze(), sample["poly"]))
        test_case.assertTrue(np.array_equal(poly_index[0][i].squeeze(), sample["poly_index"]))
