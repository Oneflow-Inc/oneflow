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
import math
import os

import cv2
import numpy as np
import oneflow as flow

VERBOSE = False
coco_dict = dict()


def _coco(anno_file):
    global coco_dict

    if anno_file not in coco_dict:
        from pycocotools.coco import COCO

        coco_dict[anno_file] = COCO(anno_file)

    return coco_dict[anno_file]


def _make_coco_data_load_fn(
    anno_file,
    image_dir,
    nthread,
    batch_size,
    stride_partition,
    shuffle_after_epoch,
    ret_image_id_only=False,
):
    flow.clear_default_session()
    flow.config.cpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def coco_load_fn():
        with flow.scope.placement("cpu", "0:0-{}".format(nthread - 1)):
            (
                image,
                image_id,
                image_size,
                gt_bbox,
                gt_label,
                gt_segm,
                gt_segm_index,
            ) = flow.data.coco_reader(
                annotation_file=anno_file,
                image_dir=image_dir,
                batch_size=batch_size,
                shuffle=shuffle_after_epoch,
                stride_partition=stride_partition,
                name="COCOReader",
            )

            if ret_image_id_only:
                return image_id

            decoded_image = flow.image_decode(image, dtype=flow.float)
            image_list = flow.tensor_buffer_to_tensor_list(
                decoded_image, shape=(800, 1333, 3), dtype=flow.float
            )
            bbox_list = flow.tensor_buffer_to_tensor_list(
                gt_bbox, shape=(128, 4), dtype=flow.float
            )
            label_list = flow.tensor_buffer_to_tensor_list(
                gt_label, shape=(128,), dtype=flow.int32
            )
            segm_list = flow.tensor_buffer_to_tensor_list(
                gt_segm, shape=(1024, 2), dtype=flow.float
            )
            segm_index_list = flow.tensor_buffer_to_tensor_list(
                gt_segm_index, shape=(1024, 3), dtype=flow.int32
            )

        return (
            image_id,
            image_size,
            image_list,
            bbox_list,
            label_list,
            segm_list,
            segm_index_list,
        )

    return coco_load_fn


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
        samples.append(
            dict(image=im, image_size=ims, bbox=b, label=l, poly=p, poly_index=pi)
        )
    return samples


def _get_category_id_to_contiguous_id_map(coco):
    return {v: i + 1 for i, v in enumerate(coco.getCatIds())}


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
            assert all([len(poly) > 0 for poly in segm]), str(
                [len(poly) for poly in segm]
            )
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


def _get_coco_sorted_imgs(anno_file):
    coco = _coco(anno_file)
    img_ids = coco.getImgIds()
    img_ids.sort()
    img_info_list = []
    for i, img_id in enumerate(img_ids):
        img_h = coco.imgs[img_id]["height"]
        img_w = coco.imgs[img_id]["width"]
        group_id = int(img_h / img_w)
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        if not _has_valid_annotation(anno):
            continue

        img_info_list.append(
            dict(index=i, image_id=img_id, group_id=group_id, anno_len=len(anno_ids))
        )

    return img_info_list


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def _has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= 10:
        return True

    return False


class GroupedDistributedSampler(object):
    def __init__(self, shards, batch_size, images, stride_sample, max_iter=3):
        assert batch_size % shards == 0
        self._images = images
        self._shards = shards
        self._shard_size = math.ceil(len(images) / shards)
        self._batch_size = batch_size
        self._batch_size_per_shard = batch_size // shards
        self._stride_sample = stride_sample
        self._max_iter = max_iter
        self._init_sample_idx()
        self._init_group_buckets()

    def _init_sample_idx(self):
        if self._stride_sample:
            self._sample_idx = list(range(self._shards))
        else:
            self._sample_idx = [rank * self._shard_size for rank in range(self._shards)]
            self._sample_idx_in_shard = [0 for _ in range(self._shards)]

    def _init_group_buckets(self):
        self._group_buckets = [[[] for _ in range(2)] for _ in range(self._shards)]

    def __iter__(self):
        for i in range(self._max_iter):
            sample_ids = []
            for rank in range(self._shards):
                sample_cnt_cur_rank = 0
                sample_ids_cur_rank = []
                group_buckets_cur_rank = self._group_buckets[rank]

                if (
                    len(group_buckets_cur_rank[0]) > 0
                    and len(group_buckets_cur_rank[1]) > 0
                ):
                    if (
                        group_buckets_cur_rank[0][0]["index"]
                        < group_buckets_cur_rank[1][0]["index"]
                    ):
                        sample = group_buckets_cur_rank[0].pop(0)
                    else:
                        sample = group_buckets_cur_rank[1].pop(0)
                elif len(group_buckets_cur_rank[0]) > 0:
                    sample = group_buckets_cur_rank[0].pop(0)
                elif len(group_buckets_cur_rank[1]) > 0:
                    sample = group_buckets_cur_rank[1].pop(0)
                else:
                    sample = self._next_sample(rank)

                group_id = sample["group_id"]
                sample_ids_cur_rank.append(sample["image_id"])
                sample_cnt_cur_rank += 1

                while sample_cnt_cur_rank < self._batch_size_per_shard:
                    if len(group_buckets_cur_rank[group_id]) > 0:
                        sample = group_buckets_cur_rank[group_id].pop(0)
                        sample_ids_cur_rank.append(sample["image_id"])
                        sample_cnt_cur_rank += 1
                        continue

                    sample = self._next_sample(rank)

                    if sample["group_id"] == group_id:
                        sample_ids_cur_rank.append(sample["image_id"])
                        sample_cnt_cur_rank += 1
                    else:
                        group_buckets_cur_rank[sample["group_id"]].append(sample)

                sample_ids.extend(sample_ids_cur_rank)

            yield sample_ids

    def _next_sample(self, rank):
        sample = self._images[self._sample_idx[rank]]
        if self._stride_sample:
            self._sample_idx[rank] += self._shards
        else:
            self._sample_idx_in_shard[rank] += 1
            self._sample_idx[rank] += 1
            if self._sample_idx_in_shard[rank] == self._shard_size:
                self._sample_idx[rank] += (self._shards - 1) * self._shard_size
                self._sample_idx_in_shard[rank] = 0

        if self._sample_idx[rank] >= len(self._images):
            self._sample_idx[rank] %= len(self._images)

        return sample


@flow.unittest.skip_unless_1n1d()
class TestCocoReader(flow.unittest.TestCase):
    def test_coco_reader(test_case, verbose=VERBOSE):
        anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
        image_dir = "/dataset/mscoco_2017/val2017"

        of_coco_load_fn = _make_coco_data_load_fn(
            anno_file, image_dir, 1, 2, True, True
        )
        (
            image_id,
            image_size,
            image,
            bbox,
            label,
            poly,
            poly_index,
        ) = of_coco_load_fn().get()
        image_id = image_id.numpy()
        image_size = image_size.numpy()
        image = image.numpy_lists()
        bbox = bbox.numpy_lists()
        label = label.numpy_lists()
        poly = poly.numpy_lists()
        poly_index = poly_index.numpy_lists()

        samples = _get_coco_image_samples(anno_file, image_dir, image_id)
        for i, sample in enumerate(samples):
            if verbose:
                print(
                    "#{} of label:\n".format(i),
                    label[0][i].squeeze(),
                    type(label[0][i].squeeze()),
                    label[0][i].squeeze().shape,
                )
                print(
                    "#{} coco label:\n".format(i),
                    sample["label"],
                    type(sample["label"]),
                    sample["label"].shape,
                )
            test_case.assertTrue(np.array_equal(image[0][i].squeeze(), sample["image"]))
            test_case.assertTrue(np.array_equal(image_size[i], sample["image_size"]))
            test_case.assertTrue(np.allclose(bbox[0][i].squeeze(), sample["bbox"]))
            cur_label = label[0][i].squeeze()
            if len(cur_label.shape) == 0:
                # when cur_label is scalar
                cur_label = np.array([cur_label])
            test_case.assertTrue(np.array_equal(cur_label, sample["label"]))
            test_case.assertTrue(np.allclose(poly[0][i].squeeze(), sample["poly"]))
            test_case.assertTrue(
                np.array_equal(poly_index[0][i].squeeze(), sample["poly_index"])
            )

    def test_coco_reader_distributed_stride(test_case, verbose=VERBOSE):
        anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
        image_dir = "/dataset/mscoco_2017/val2017"

        image_info_list = _get_coco_sorted_imgs(anno_file)
        if verbose:
            print("Info of the first 20 images:")
            for i, image_info in enumerate(image_info_list[:20]):
                print(
                    "index: {}, image_id: {}, group_id: {}, anno len: {}".format(
                        i,
                        image_info["image_id"],
                        image_info["group_id"],
                        image_info["anno_len"],
                    )
                )

        sampler = GroupedDistributedSampler(4, 8, image_info_list, True)
        of_coco_load_fn = _make_coco_data_load_fn(
            anno_file, image_dir, 4, 8, True, False, True
        )
        for i, sample_ids in enumerate(sampler):
            image_id = of_coco_load_fn().get().numpy()
            if verbose:
                print("#{} image_id:".format(i), image_id)
                print("#{} sample_ids:".format(i), sample_ids)
            test_case.assertTrue(np.array_equal(image_id, sample_ids))

    def test_coco_reader_distributed_contiguous(test_case, verbose=VERBOSE):
        anno_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
        image_dir = "/dataset/mscoco_2017/val2017"

        image_info_list = _get_coco_sorted_imgs(anno_file)
        sampler = GroupedDistributedSampler(4, 8, image_info_list, False)
        of_coco_load_fn = _make_coco_data_load_fn(
            anno_file, image_dir, 4, 8, False, False, True
        )
        for i, sample_ids in enumerate(sampler):
            image_id = of_coco_load_fn().get().numpy()
            if verbose:
                print("#{} image_id:".format(i), image_id)
                print("#{} sample_ids:".format(i), sample_ids)
            test_case.assertTrue(np.array_equal(image_id, sample_ids))


if __name__ == "__main__":
    unittest.main()
