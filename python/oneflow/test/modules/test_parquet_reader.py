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
import os
import numpy as np
import copy

import oneflow as flow
import oneflow.unittest


def make_parquet_reader(
    path,
    schema,
    batch_size=256,
    shuffle=True,
    completely_shuffle=False,
    random_seed=12345,
    prefetch_buffer_size=None,
    read_footprint=1024,
    use_mmap=True,
    device=None,
    placement=None,
    sbp=None,
):
    return flow.nn.ParquetReader(
        path=path,
        schema=schema,
        batch_size=batch_size,
        shuffle=shuffle,
        completely_shuffle=completely_shuffle,
        random_seed=random_seed,
        prefetch_buffer_size=prefetch_buffer_size,
        read_footprint=read_footprint,
        use_mmap=use_mmap,
        device=device,
        placement=placement,
        sbp=sbp,
    )


class OFRecordReader(flow.nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_ofrecord/ofrecord",
        data_part_num: int = 256,
        part_name_suffix_length: int = 5,
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = False,
        device=None,
        placement=None,
        sbp=None,
    ):
        super(OFRecordReader, self).__init__()
        assert mode in ("train", "val")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.mode = mode

        self.reader = flow.nn.OfrecordReader(
            os.path.join(data_dir, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=shuffle,
            shuffle_after_epoch=shuffle,
            device=device,
            placement=placement,
            sbp=sbp,
        )

        def _blob_decoder(bn, shape, dtype=flow.int32):
            return flow.nn.OfrecordRawDecoder(bn, shape=shape, dtype=dtype)

        self.labels = _blob_decoder("labels", (1,), flow.float)
        self.dense_fields = _blob_decoder(
            "dense_fields", (num_dense_fields,), flow.float
        )
        self.sparse_fields = _blob_decoder("deep_sparse_fields", (num_sparse_fields,))

    def forward(self):
        reader = self.reader()
        labels = self.labels(reader)
        dense_fields = self.dense_fields(reader)
        sparse_fields = self.sparse_fields(reader)
        return labels, dense_fields, sparse_fields


class MyGraph(flow.nn.Graph):
    def __init__(self, reader):
        super().__init__()
        self.reader_ = reader

    def build(self):
        return self.reader_()


def _test_fixed_length_columns_shuffle_determinism(test_case, completely_shuffle):
    parquet_dir = "/dataset/wdl_parquet/train/"
    schema = [
        # {"col_id": 0, "shape": (26,), "dtype": flow.int32},
        {"col_id": 1, "shape": (13,), "dtype": flow.float32},
        # {"col_id": 2, "shape": (), "dtype": flow.int32},
        {"col_name": "labels", "shape": (), "dtype": flow.int32},
        # {"col_id": 3, "shape": (2,), "dtype": flow.int32},
    ]

    reader = make_parquet_reader(
        parquet_dir,
        copy.deepcopy(schema),
        batch_size=256,
        shuffle=True,
        completely_shuffle=completely_shuffle,
        random_seed=12345,
    )
    results = reader()

    reader_ = make_parquet_reader(
        parquet_dir,
        schema,
        batch_size=256,
        shuffle=True,
        completely_shuffle=completely_shuffle,
        random_seed=12345,
    )
    results_ = reader_()

    for (data, data_) in zip(results, results_):
        test_case.assertTrue(np.allclose(data.numpy(), data_.numpy()))


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class ParquetReaderTestCase(oneflow.unittest.TestCase):
    def test_compare_with_ofrecord(test_case):
        parquet_file = "/dataset/dlrm_parquet/val/part-00000-9aa77b80-babb-496b-908e-457d9f48bb06-c000.snappy.parquet"
        schema = [
            {"col_id": 0, "shape": (), "dtype": flow.double},
            {"col_id": 1, "shape": (13,), "dtype": flow.double},
            {"col_id": 2, "shape": (26,), "dtype": flow.int32},
        ]

        parquet_reader = make_parquet_reader(
            parquet_file, schema, batch_size=20, shuffle=False
        )

        labels, dense_fields, sparse_fields = parquet_reader()

        ofrecord_file_dir = "/dataset/dlrm_ofrecord/"
        ofrecord_reader = OFRecordReader(
            data_dir=ofrecord_file_dir, batch_size=20, shuffle=False, mode="val"
        )
        labels_, dense_fields_, sparse_fields_ = ofrecord_reader()

        test_case.assertTrue(np.allclose(labels.numpy(), labels_.numpy().ravel()))
        test_case.assertTrue(np.allclose(dense_fields.numpy(), dense_fields_.numpy()))
        test_case.assertTrue(np.allclose(sparse_fields.numpy(), sparse_fields_.numpy()))

    def test_fixed_length_columns_shuffle_determinism(test_case):
        _test_fixed_length_columns_shuffle_determinism(test_case, False)
        _test_fixed_length_columns_shuffle_determinism(test_case, True)

    @unittest.skipIf(True, "")
    def test_imagenet(test_case):
        pass


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class DistributedParquetReaderTestCase(oneflow.unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
