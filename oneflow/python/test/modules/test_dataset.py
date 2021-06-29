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
import os
import unittest
import cv2
import numpy as np
import oneflow.experimental as flow
from oneflow.python.test.ops.test_coco_reader import _get_coco_image_samples


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestOFRecordModule(flow.unittest.TestCase):
    def test_record(test_case):
        flow.InitEagerGlobalSession()
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
        image_raw_buffer_nd = image_raw_buffer.numpy()[0]

        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_tensor_buffer_image.png")
        test_case.assertTrue(np.array_equal(image_raw_buffer_nd, gt_np))

        image = resize(image_raw_buffer)

        resized_image_raw_buffer_nd = image.numpy()[0]
        gt_np = cv2.imread(
            "/dataset/imagenette/ofrecord/gt_tensor_buffer_resized_image.png"
        )
        test_case.assertTrue(np.array_equal(resized_image_raw_buffer_nd, gt_np))

        image = crop_mirror_normal(image)

        # recover image
        image_np = image.numpy()
        image_np = np.squeeze(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * rgb_std + rgb_mean
        image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.uint8)

        # read gt
        gt_np = cv2.imread("/dataset/imagenette/ofrecord/gt_val_image.png")

        test_case.assertEqual(label.numpy()[0], 5)
        test_case.assertTrue(np.array_equal(image_np, gt_np))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCOCOModule(flow.unittest.TestCase):
    def test_coco(test_case, verbose=True):
        batch_size = 1
        annotation_file = (
            "/dataset/mscoco_2017/annotations/sample_10_instances_val2017.json"
        )
        image_dir = "/dataset/mscoco_2017/val2017"

        coco_loader = flow.nn.COCOReader(
            annotation_file=annotation_file,
            image_dir=image_dir,
            batch_size=batch_size,
            shuffle=True,
            random_seed=-1,
            group_by_aspect_ratio=True,
            remove_images_without_annotations=True,
            stride_partition=True,
        )
        (
            image,
            image_id,
            image_size,
            gt_bbox,
            gt_label,
            gt_segm,
            gt_segm_index,
        ) = coco_loader()
        print(type(image), type(gt_bbox))
        coco_image_decoder = flow.nn.COCOImageDecoder(dtype=flow.float)

        decoded_image = coco_image_decoder(image)
        x = flow.gen_tensor_buffer(
            (2, 2),
            [(10, 10), (50, 50), (20, 80), (100, 100)],
            [0.0, 1.0, 2.0, 3.0],
            flow.float32,
        )

        # y = flow.tensor_buffer_to_list_of_tensors(
        #     x, [(100, 100)], [flow.float32]
        # )

        # decoded_image.numpy()
        # shapes = decoded_image._tensor_buffer_shapes_and_dtypes
        image_list = flow.tensor_buffer_to_list_of_tensors(
            decoded_image, [(800, 1333, 3)], [flow.float]
        )
        bbox_list = flow.tensor_buffer_to_list_of_tensors(
            gt_bbox, [(128, 4)], [flow.float]
        )
        label_list = flow.tensor_buffer_to_list_of_tensors(
            gt_label, [(128,)], [flow.int32]
        )
        segm_list = flow.tensor_buffer_to_list_of_tensors(
            gt_segm, [(1024, 2)], [flow.float]
        )
        segm_index_list = flow.tensor_buffer_to_list_of_tensors(
            gt_segm_index, [(1024, 3)], [flow.int32]
        )

        print(len(image))
        image = image_list.numpy_lists()
        image_id = image_id.numpy()
        image_size = image_size.numpy()
        bbox = bbox_list.numpy_lists()
        label = label_list.numpy_lists()
        poly = segm_list.numpy_lists()
        poly_index = segm_index_list.numpy_lists()

        samples = _get_coco_image_samples(annotation_file, image_dir, image_id)
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
                test_case.assertTrue(
                    np.array_equal(image[0][i].squeeze(), sample["image"])
                )
                test_case.assertTrue(
                    np.array_equal(image_size[i], sample["image_size"])
                )
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


if __name__ == "__main__":
    unittest.main()
