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

import math
import os
import unittest

import cv2
import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestOFRecordModule(flow.unittest.TestCase):
    def test_record(test_case):
        batch_size = 1
        color_space = "RGB"
        height = 224
        width = 224
        output_layout = "NCHW"
        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        record_reader = flow.nn.OFRecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            shuffle_after_epoch=False,
        )
        record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        record_label_decoder = flow.nn.OFRecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )
        resize = flow.nn.image.Resize(
            resize_side="shorter", keep_aspect_ratio=True, target_size=256
        )
        crop_mirror_normal = flow.nn.CropMirrorNormalize(
            color_space=color_space,
            output_layout=output_layout,
            crop_h=height,
            crop_w=width,
            crop_pos_y=0.5,
            crop_pos_x=0.5,
            mean=rgb_mean,
            std=rgb_std,
            output_dtype=flow.float,
        )
        val_record = record_reader()
        label = record_label_decoder(val_record)
        image_raw_buffer = record_image_decoder(val_record)
        image_raw_buffer_nd = image_raw_buffer.numpy()
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_tensor_buffer_image.png")
        test_case.assertTrue(np.array_equal(image_raw_buffer_nd[0], gt_np))
        image = resize(image_raw_buffer)[0]
        resized_image_raw_buffer_nd = image.numpy()
        gt_np = cv2.imread(
            "/dataset/imagenette/ofrecord/gt_tensor_buffer_resized_image.png"
        )
        test_case.assertTrue(np.array_equal(resized_image_raw_buffer_nd[0], gt_np))
        image = crop_mirror_normal(image)
        image_np = image.numpy()
        image_np = np.squeeze(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * rgb_std + rgb_mean
        image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.uint8)
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_val_image.png")
        test_case.assertEqual(label.numpy(), 5)
        test_case.assertTrue(np.array_equal(image_np, gt_np))


@flow.unittest.skip_unless_1n1d()
class TestGlobalOFRecordModule(flow.unittest.TestCase):
    def test_global_record(test_case):
        batch_size = 1
        color_space = "RGB"
        height = 224
        width = 224
        output_layout = "NCHW"
        rgb_mean = [123.68, 116.779, 103.939]
        rgb_std = [58.393, 57.12, 57.375]
        record_reader = flow.nn.OfrecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            shuffle_after_epoch=False,
            placement=flow.placement("cpu", ranks=[0]),
            sbp=[flow.sbp.split(0)],
        )
        record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        record_label_decoder = flow.nn.OfrecordRawDecoder(
            "class/label", shape=(), dtype=flow.int32
        )
        resize = flow.nn.image.Resize(
            resize_side="shorter", keep_aspect_ratio=True, target_size=256
        )
        flip = flow.nn.CoinFlip(
            batch_size=batch_size,
            placement=flow.placement("cpu", ranks=[0]),
            sbp=[flow.sbp.split(0)],
        )
        crop_mirror_normal = flow.nn.CropMirrorNormalize(
            color_space=color_space,
            output_layout=output_layout,
            crop_h=height,
            crop_w=width,
            crop_pos_y=0.5,
            crop_pos_x=0.5,
            mean=rgb_mean,
            std=rgb_std,
            output_dtype=flow.float,
        )
        rng = flip()
        val_record = record_reader()
        label = record_label_decoder(val_record)
        image_raw_buffer = record_image_decoder(val_record)
        image_raw_buffer_nd = image_raw_buffer.to_local().numpy()
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_tensor_buffer_image.png")
        test_case.assertTrue(np.array_equal(image_raw_buffer_nd[0], gt_np))
        image = resize(image_raw_buffer)[0]
        resized_image_raw_buffer_nd = image.to_local().numpy()
        gt_np = cv2.imread(
            "/dataset/imagenette/ofrecord/gt_tensor_buffer_resized_image.png"
        )
        test_case.assertTrue(np.array_equal(resized_image_raw_buffer_nd[0], gt_np))
        image = crop_mirror_normal(image)
        image_np = image.to_local().numpy()
        image_np = np.squeeze(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * rgb_std + rgb_mean
        image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.uint8)
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_val_image.png")
        test_case.assertEqual(label.to_local().numpy(), 5)
        test_case.assertTrue(np.array_equal(image_np, gt_np))


coco_dict = dict()


def _coco(anno_file):
    global coco_dict
    if anno_file not in coco_dict:
        from pycocotools.coco import COCO

        coco_dict[anno_file] = COCO(anno_file)
    return coco_dict[anno_file]


def _get_coco_image_samples(anno_file, image_dir, image_ids):
    coco = _coco(anno_file)
    category_id_to_contiguous_id_map = _get_category_id_to_contiguous_id_map(coco)
    (image, image_size) = _read_images_with_cv(coco, image_dir, image_ids)
    bbox = _read_bbox(coco, image_ids)
    label = _read_label(coco, image_ids, category_id_to_contiguous_id_map)
    img_segm_poly_list = _read_segm_poly(coco, image_ids)
    (poly, poly_index) = _segm_poly_list_to_tensor(img_segm_poly_list)
    samples = []
    for (im, ims, b, l, p, pi) in zip(image, image_size, bbox, label, poly, poly_index):
        samples.append(
            dict(image=im, image_size=ims, bbox=b, label=l, poly=p, poly_index=pi)
        )
    return samples


def _get_category_id_to_contiguous_id_map(coco):
    return {v: i + 1 for (i, v) in enumerate(coco.getCatIds())}


def _read_images_with_cv(coco, image_dir, image_ids):
    image_files = [
        os.path.join(image_dir, coco.imgs[img_id]["file_name"]) for img_id in image_ids
    ]
    image_size = [
        (coco.imgs[img_id]["height"], coco.imgs[img_id]["width"])
        for img_id in image_ids
    ]
    return (
        [cv2.imread(image_file).astype(np.single) for image_file in image_files],
        image_size,
    )


def _bbox_convert_from_xywh_to_xyxy(bbox, image_h, image_w):
    (x, y, w, h) = bbox
    (x1, y1) = (x, y)
    x2 = x1 + max(w - 1, 0)
    y2 = y1 + max(h - 1, 0)
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
            assert all([len(poly) > 0 for poly in segm]), str(
                [len(poly) for poly in segm]
            )
            segm_poly_list.append(segm)
        img_segm_poly_list.append(segm_poly_list)
    return img_segm_poly_list


def _segm_poly_list_to_tensor(img_segm_poly_list):
    poly_array_list = []
    poly_index_array_list = []
    for (img_idx, segm_poly_list) in enumerate(img_segm_poly_list):
        img_poly_elem_list = []
        img_poly_index_list = []
        for (obj_idx, poly_list) in enumerate(segm_poly_list):
            for (poly_idx, poly) in enumerate(poly_list):
                img_poly_elem_list.extend(poly)
                for (pt_idx, pt) in enumerate(poly):
                    if pt_idx % 2 == 0:
                        img_poly_index_list.append([pt_idx / 2, poly_idx, obj_idx])
        img_poly_array = np.array(img_poly_elem_list, dtype=np.single).reshape(-1, 2)
        assert img_poly_array.size > 0, segm_poly_list
        poly_array_list.append(img_poly_array)
        img_poly_index_array = np.array(img_poly_index_list, dtype=np.int32)
        assert img_poly_index_array.size > 0, segm_poly_list
        poly_index_array_list.append(img_poly_index_array)
    return (poly_array_list, poly_index_array_list)


@flow.unittest.skip_unless_1n1d()
class TestCocoReader(flow.unittest.TestCase):
    def test_coco_reader(test_case):
        anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
        image_dir = "/dataset/mscoco_2017/val2017"
        num_iterations = 10
        coco_reader = flow.nn.COCOReader(
            annotation_file=anno_file,
            image_dir=image_dir,
            batch_size=2,
            shuffle=True,
            stride_partition=True,
        )
        image_decoder = flow.nn.image.decode(dtype=flow.float)
        for i in range(num_iterations):
            (
                image,
                image_id,
                image_size,
                gt_bbox,
                gt_label,
                gt_segm,
                gt_segm_index,
            ) = coco_reader()
            decoded_image = image_decoder(image)
            image_list = decoded_image.numpy()
            image_id = image_id.numpy()
            image_size = image_size.numpy()
            bbox_list = gt_bbox.numpy()
            label_list = gt_label.numpy()
            segm_list = gt_segm.numpy()
            segm_index_list = gt_segm_index.numpy()
            samples = _get_coco_image_samples(anno_file, image_dir, image_id)
            for (i, sample) in enumerate(samples):
                test_case.assertTrue(np.array_equal(image_list[i], sample["image"]))
                test_case.assertTrue(
                    np.array_equal(image_size[i], sample["image_size"])
                )
                test_case.assertTrue(np.allclose(bbox_list[i], sample["bbox"]))
                cur_label = label_list[i]
                if len(cur_label.shape) == 0:
                    cur_label = np.array([cur_label])
                test_case.assertTrue(np.array_equal(cur_label, sample["label"]))
                test_case.assertTrue(np.allclose(segm_list[i], sample["poly"]))
                test_case.assertTrue(
                    np.array_equal(segm_index_list[i], sample["poly_index"])
                )


@flow.unittest.skip_unless_1n1d()
class TestOFRecordBytesDecoder(flow.unittest.TestCase):
    def test_OFRecordBytesDecoder(test_case):
        batch_size = 16
        record_reader = flow.nn.OFRecordReader(
            "/dataset/imagenette/ofrecord",
            batch_size=batch_size,
            part_name_suffix_length=5,
        )
        val_record = record_reader()

        bytesdecoder_img = flow.nn.OFRecordBytesDecoder("encoded")

        image_raw_buffer = bytesdecoder_img(val_record)

        image_raw_buffer_nd = image_raw_buffer.numpy()[0]
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_tensor_buffer_image.png")
        img = cv2.imdecode(image_raw_buffer_nd, cv2.IMREAD_COLOR)
        test_case.assertTrue(np.array_equal(img, gt_np))


if __name__ == "__main__":
    unittest.main()
