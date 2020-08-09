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
import oneflow as flow
import oneflow.typing as tp

BATCH = 2
HEIGHT = 300
WIDTH = 300
NUM_BOXES = 5


def RunSsdRandomCropJob(
    image_list,
    bboxes_list,
    labels_list,
    iou_overlap_ranges=((0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0)),
    size_shrink_rate=((0.316, 1.0), (0.316, 1.0)),
    aspect_ratio_range=(0.5, 2.0),
    random_seed=None,
    max_num_attempt=100,
):
    flow.config.enable_debug_mode(True)
    flow.clear_default_session()
    func_config = flow.function_config()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def ssd_rand_crop_fn(
        image: tp.ListListNumpy.Placeholder(
            shape=(BATCH, HEIGHT, WIDTH, 3), dtype=flow.float
        ),
        bboxes: tp.ListListNumpy.Placeholder(
            shape=(BATCH, NUM_BOXES, 4), dtype=flow.float
        ),
        labels: tp.ListListNumpy.Placeholder(
            shape=(BATCH, NUM_BOXES,), dtype=flow.int32
        ),
    ):
        image_buffer = flow.tensor_list_to_tensor_buffer(image)
        bboxes_buffer = flow.tensor_list_to_tensor_buffer(bboxes)
        labels_buffer = flow.tensor_list_to_tensor_buffer(labels)
        with flow.scope.placement("cpu", "0:0"):
            (new_image, new_bboxes, new_labels,) = flow.ssd_random_crop(
                image_buffer,
                bbox=bboxes_buffer,
                label=labels_buffer,
                iou_overlap_ranges=iou_overlap_ranges,
                size_shrink_rates=size_shrink_rate,
                aspect_ratio_range=aspect_ratio_range,
                random_seed=random_seed,
                max_num_attempts=max_num_attempt,
            )
            new_image = flow.tensor_buffer_to_tensor_list(
                new_image, shape=(HEIGHT, WIDTH, 3), dtype=flow.float
            )
            new_bboxes = flow.tensor_buffer_to_tensor_list(
                new_bboxes, shape=(NUM_BOXES, 4), dtype=flow.float
            )
            new_labels = flow.tensor_buffer_to_tensor_list(
                new_labels, shape=(NUM_BOXES,), dtype=flow.int32
            )
            return (new_image, new_bboxes, new_labels)

    crop_image_tensor, crop_bboxes_tensor, crop_labels_tensor = ssd_rand_crop_fn(
        [image_list], [bboxes_list], [labels_list]
    ).get()
    crop_image_list = crop_image_tensor.numpy_lists()[0]
    crop_bboxes_list = crop_bboxes_tensor.numpy_lists()[0]
    crop_labels_list = crop_labels_tensor.numpy_lists()[0]

    area_ratio = []
    aspect_ratio_list = []
    size_shrink_rate_np = np.array(size_shrink_rate, dtype=np.float32)
    area_range = (size_shrink_rate_np[0] * size_shrink_rate_np[1]).tolist()
    for crop_image, bboxes, labels, image, orig_bboxes in zip(
        crop_image_list, crop_bboxes_list, crop_labels_list, image_list, bboxes_list
    ):
        crop_height = crop_image.shape[1]
        crop_width = crop_image.shape[2]
        aspect_ratio = crop_height / crop_width
        aspect_ratio_list.append(aspect_ratio)
        area = crop_width * crop_height
        original_area = image.shape[1] * image.shape[2]
        area_ratio.append(area / original_area)
        assert bboxes.shape[1] == labels.shape[1]
        assert np.max(bboxes[0, :, 2]) <= crop_width
        assert np.max(bboxes[0, :, 3]) <= crop_height
        assert np.min(bboxes[0, :, 0]) >= 0.0
        assert np.min(bboxes[0, :, 1]) >= 0.0
    assert np.greater_equal(min(aspect_ratio_list), aspect_ratio_range[0])
    assert np.less_equal(max(aspect_ratio_list), aspect_ratio_range[1])
    assert np.greater_equal(min(area_ratio), area_range[0])
    assert np.less_equal(max(area_ratio), area_range[1])


def random_bbox():
    x1 = np.random.uniform(low=0.0, high=WIDTH, size=NUM_BOXES).astype(np.float32)
    x2 = np.random.uniform(low=x1, high=WIDTH, size=NUM_BOXES).astype(np.float32)
    y1 = np.random.uniform(low=0.0, high=HEIGHT, size=NUM_BOXES).astype(np.float32)
    y2 = np.random.uniform(low=y1, high=HEIGHT, size=NUM_BOXES).astype(np.float32)
    return np.stack((x1, y1, x2, y2), axis=1).reshape(1, NUM_BOXES, 4)


def test_whole_image_bbox(test_case):
    image_size = [1, HEIGHT, WIDTH, 3]
    image_list = [
        np.arange(0, np.prod(image_size), dtype=np.float32).reshape(image_size)
        for _ in range(BATCH)
    ]
    bbox = np.tile(
        np.array([0.0, 0.0, WIDTH, HEIGHT], dtype=np.float32), (NUM_BOXES, 1)
    ).reshape(1, NUM_BOXES, 4)
    bboxes_list = [bbox for _ in range(BATCH)]
    labels_list = [
        np.random.randint(low=1, high=81, size=(1, NUM_BOXES,), dtype=np.int32)
        for _ in range(BATCH)
    ]
    RunSsdRandomCropJob(image_list, bboxes_list, labels_list)


def test_random_bbox(test_case):
    image_size = [1, HEIGHT, WIDTH, 3]
    image_list = [
        np.arange(0, np.prod(image_size), dtype=np.float32).reshape(image_size)
        for _ in range(BATCH)
    ]
    bboxes_list = [random_bbox() for _ in range(BATCH)]
    labels_list = [
        np.random.randint(low=1, high=81, size=(1, NUM_BOXES,), dtype=np.int32)
        for _ in range(BATCH)
    ]
    RunSsdRandomCropJob(image_list, bboxes_list, labels_list)
