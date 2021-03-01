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
import random
import os
import struct
import cv2
import numpy as np
import oneflow.core.record.record_pb2 as record_pb


class OFRecordDataset(object):
    def __init__(
        self,
        data_dir,
        num_data_parts,
        part_name_suffix_length,
        batch_size,
        shuffle_data_part,
    ):
        self.data_dir_ = data_dir
        self.num_data_parts_ = num_data_parts
        self.part_name_suffix_length_ = part_name_suffix_length
        self.batch_size_ = batch_size
        self.epoch_cnt_ = 0
        self.cur_data_part_idx_ = 0
        self.shuffle_data_part_ = shuffle_data_part
        self.reader_ = None
        self.num_read_batchs_ = 0

    @property
    def batch_size(self):
        return self.batch_size_

    @batch_size.setter
    def batch_size(self, bs):
        self.batch_size_ = bs

    @property
    def num_read_batchs(self):
        return self.num_read_batchs_

    def __del__(self):
        if self.reader_ is not None:
            self.reader_.close()

    def __iter__(self):
        self._gen_data_part_seq()
        self._open_data_part_file()

        while True:
            yield self._read_one_batch()

    def load_batchs(self, num_batchs):
        image_list = []
        label_list = []

        for i, (image_array, label_array) in enumerate(self):
            if i >= num_batchs:
                break

            image_list.append(image_array)
            label_list.append(label_array)

        return image_list, label_list

    def parse_record(self, record):
        raise NotImplementedError

    def collate(self, batch):
        raise NotImplementedError

    def reset(self):
        self.epoch_cnt_ = 0
        self.cur_data_part_idx_ = 0
        if self.reader_ is not None:
            self.reader_.close()
        self.num_read_batchs_ = 0

    def _move_to_next_data_part(self):
        self.cur_data_part_idx_ += 1
        if self.cur_data_part_idx_ >= len(self.data_part_seq_):
            self.epoch_cnt_ += 1
            self._gen_data_part_seq()

        self._open_data_part_file()

    def _gen_data_part_seq(self):
        data_part_name_pattern = (
            r"part-{:0" + str(self.part_name_suffix_length_) + r"d}"
        )
        self.data_part_seq_ = [
            data_part_name_pattern.format(i) for i in range(self.num_data_parts_)
        ]
        if self.shuffle_data_part_:
            random.shuffle(self.data_part_seq_)

    def _open_data_part_file(self):
        if self.reader_ is not None:
            self.reader_.close()

        data_part_file_path = os.path.join(
            self.data_dir_, self.data_part_seq_[self.cur_data_part_idx_]
        )
        self.reader_ = open(data_part_file_path, "rb")

    def _read_one_batch(self):
        assert self.reader_ is not None

        batch = []
        for i in range(self.batch_size_):
            record_head = self.reader_.read(8)
            if record_head is None or len(record_head) != 8:
                self._move_to_next_data_part()
                break

            record = record_pb.OFRecord()
            record_byte_size = struct.unpack("q", record_head)[0]
            record.ParseFromString(self.reader_.read(record_byte_size))
            batch.append(self.parse_record(record))

        self.num_read_batchs_ += 1
        return self.collate(batch)


class ImageNetRecordDataset(OFRecordDataset):
    def __init__(
        self,
        data_dir="/dataset/ImageNet/ofrecord/validation",
        num_data_parts=256,
        part_name_suffix_length=5,
        batch_size=4,
        shuffle_data_part=False,
        image_resize_size=224,
        data_format="NCHW",
    ):
        super().__init__(
            data_dir,
            num_data_parts,
            part_name_suffix_length,
            batch_size,
            shuffle_data_part,
        )
        self.image_resize_size_ = image_resize_size
        self.data_format_ = data_format

    def parse_record(self, record):
        image_raw_bytes = record.feature["encoded"].bytes_list.value[0]
        image = cv2.imdecode(
            np.frombuffer(image_raw_bytes, np.uint8), cv2.IMREAD_COLOR
        ).astype(np.float32)
        image = self.preprocess_image(image)
        label = record.feature["class/label"].int32_list.value[0]
        return (image, label)

    def collate(self, batch):
        batched_image = np.stack([data[0] for data in batch], axis=0)
        batched_label = np.array([data[1] for data in batch], dtype=np.int32)
        return batched_image, batched_label

    def preprocess_image(self, image):
        # resize
        image = cv2.resize(image, (self.image_resize_size_, self.image_resize_size_))
        # bgr to rgb (opencv decoded image is bgr format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalize
        norm_rgb_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        norm_rgb_std = np.array([58.393, 57.12, 57.375], dtype=np.float32)
        image = (image - norm_rgb_mean) / norm_rgb_std
        # NHWC to NCHW
        if self.data_format_ == "NCHW":
            assert image.shape[2] == 3
            image = np.transpose(image, (2, 0, 1))
        elif self.data_format_ == "NHWC":
            assert image.shape[2] == 3
        else:
            raise ValueError("Unsupported image data format")

        return np.ascontiguousarray(image)


class FaceEmoreRecordDataset(OFRecordDataset):
    def __init__(
        self,
        data_dir="/dataset/insightface/train_ofrecord/faces_emore",
        num_data_parts=256,
        part_name_suffix_length=1,
        batch_size=4,
        shuffle_data_part=False,
        image_width=112,
        image_height=112,
        color_space="RGB",
        data_format="NCHW",
    ):
        super().__init__(
            data_dir,
            num_data_parts,
            part_name_suffix_length,
            batch_size,
            shuffle_data_part,
        )
        self.image_width_ = image_width
        self.image_height_ = image_height
        self.color_space_ = color_space
        self.data_format_ = data_format

    def parse_record(self, record):
        image_raw_bytes = record.feature["encoded"].bytes_list.value[0]
        image = cv2.imdecode(
            np.frombuffer(image_raw_bytes, np.uint8), cv2.IMREAD_COLOR
        ).astype(np.float32)
        image = self.preprocess_image(image)
        issame = record.feature["issame"].int32_list.value[0]
        return (image, issame)

    def preprocess_image(self, image):
        # resize
        image = cv2.resize(image, (self.image_height_, self.image_width_))
        # bgr to rgb (opencv decoded image is bgr format)
        if self.color_space_ == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # NHWC to NCHW
        if self.data_format_ == "NCHW":
            assert image.shape[2] == 3
            image = np.transpose(image, (2, 0, 1))
        elif self.data_format_ == "NHWC":
            assert image.shape[2] == 3
        else:
            raise ValueError("Unsupported image data format")
        return image

    def collate(self, batch):
        image = np.stack([data[0] for data in batch], axis=0)
        issame = np.array([data[1] for data in batch], dtype=np.int32)
        return image, issame
