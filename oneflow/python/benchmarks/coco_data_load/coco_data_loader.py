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
import oneflow as flow
import math
import pandas as pd
import time


class COCODataLoadConfig(object):
    def __init__(self):
        self.annotation_file = (
            "/dataset/mscoco_2017/annotations/instances_train2017.json"
        )
        self.image_dir = "/dataset/mscoco_2017/train2017"
        # self.annotation_file = "/dataset/mscoco_2017/annotations/instances_val2017.json"
        # self.image_dir = "/dataset/mscoco_2017/val2017"
        self.shuffle_after_epoch = True
        self.stride_partition = False
        self.batch_size = 2
        self.target_size = 800
        self.max_size = 1333
        self.image_align_size = 32
        self.image_normal_std = (1.0, 1.0, 1.0)
        self.image_normal_mean = (102.9801, 115.9465, 122.7717)
        self.max_num_objs = 512


def roundup(x, align):
    return int(math.ceil(x / float(align)) * align)


def coco_data_load(cfg, machine_id, nrank):
    with flow.scope.placement("cpu", "{}:0-{}".format(machine_id, nrank - 1)):
        (
            image,
            image_id,
            image_size,
            bbox,
            label,
            segm_poly,
            segm_poly_index,
        ) = flow.data.coco_reader(
            annotation_file=cfg.annotation_file,
            image_dir=cfg.image_dir,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle_after_epoch,
            stride_partition=cfg.stride_partition,
        )
        # image decode
        image = flow.image.decode(image, dtype=flow.float)
        # image target resize
        aligned_target_size = roundup(cfg.target_size, cfg.image_align_size)
        aligned_max_size = roundup(cfg.max_size, cfg.image_align_size)
        image, new_size, scale = flow.image.target_resize(
            image, target_size=aligned_target_size, max_size=aligned_max_size
        )
        bbox = flow.detection.object_bbox_scale(bbox, scale)
        segm_poly = flow.detection.object_segmentation_polygon_scale(segm_poly, scale)
        # random flip
        flip_code = flow.random.coin_flip(cfg.batch_size)
        image = flow.image.flip(image, flip_code)
        bbox = flow.detection.object_bbox_flip(bbox, new_size, flip_code)
        segm_poly = flow.detection.object_segmentation_polygon_flip(
            segm_poly, new_size, flip_code
        )
        # image normalize
        image = flow.image.normalize(image, cfg.image_normal_std, cfg.image_normal_mean)
        # batch collate
        image = flow.image.batch_align(
            image,
            shape=(aligned_target_size, aligned_max_size, 3),
            dtype=flow.float,
            alignment=cfg.image_align_size,
        )
        # TODO(zhangwenxiao, jiangxuefei): refine in multi-client
        # gt_bbox = flow.tensor_buffer_to_tensor_list(
        #     bbox, shape=(cfg.max_num_objs, 4), dtype=flow.float
        # )
        # gt_label = flow.tensor_buffer_to_tensor_list(
        #     label, shape=(cfg.max_num_objs,), dtype=flow.int32
        # )
        # segm_mask = flow.detection.object_segmentation_polygon_to_mask(
        #     segm_poly, segm_poly_index, new_size
        # )
        # gt_mask = flow.tensor_buffer_to_tensor_list(
        #     segm_mask,
        #     shape=(cfg.max_num_objs, aligned_target_size, aligned_max_size),
        #     dtype=flow.int8,
        # )
        return image, new_size, gt_bbox, gt_label, gt_mask


def _make_data_load_fn():
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    cfg = COCODataLoadConfig()

    @flow.global_function(func_config)
    def data_load_fn():
        return coco_data_load(cfg, 0, 1)

    return data_load_fn


def _benchmark(iter_num, drop_first_iters, verbose=False):
    flow.env.init()
    data_loader = _make_data_load_fn()
    s = pd.Series([], name="time_elapsed", dtype="float32")
    timestamp = time.perf_counter()
    for i in range(iter_num):
        # data_loader().get()
        image, image_size, gt_bbox, gt_label, gt_mask = data_loader().get()
        cur = time.perf_counter()
        s[i] = cur - timestamp
        timestamp = cur

        if verbose:
            print("==== iter {} ====".format(i))
            print(
                "image: {}\n".format(image.numpy_list()[0].shape),
                image.numpy_list()[0],
            )
            print(
                "image_size: {}\n".format(image_size.numpy().shape), image_size.numpy(),
            )
            print("gt_bbox:\n", gt_bbox.numpy_lists()[0])
            print("gt_label:\n", gt_label.numpy_lists()[0])
            print("gt_mask:\n", gt_mask.numpy_lists()[0])

    print(
        "mean of time elapsed of {} iters (dropped {} first iters): {}".format(
            iter_num, drop_first_iters, s[drop_first_iters:].mean()
        )
    )
    s.to_csv("coco_data_benchmark.csv", header=True)


if __name__ == "__main__":
    _benchmark(500, 10)
