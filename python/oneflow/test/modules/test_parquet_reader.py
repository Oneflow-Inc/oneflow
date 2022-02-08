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
import time

import oneflow as flow
import oneflow.unittest


def make_parquet_reader(
    path,
    schema,
    batch_size=256,
    shuffle=True,
    completely_shuffle=False,
    random_seed=12345,
    shuffle_buffer_size=32,
    prefetch_buffer_size=4,
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
        shuffle_buffer_size=shuffle_buffer_size,
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
    parquet_dir = "/dataset/wdl_parquet/train"
    schema = [
        {"col_id": 0, "shape": (26,), "dtype": flow.int32},
        {"col_id": 1, "shape": (13,), "dtype": flow.float32},
        {"col_name": "labels", "shape": (), "dtype": flow.int32},
        {"col_id": 3, "shape": (2,), "dtype": flow.int32},
    ]
    batch_size = 16

    reader = make_parquet_reader(
        parquet_dir,
        copy.deepcopy(schema),
        batch_size=batch_size,
        shuffle=True,
        completely_shuffle=completely_shuffle,
        random_seed=12345,
    )

    reader_ = make_parquet_reader(
        parquet_dir,
        schema,
        batch_size=batch_size,
        shuffle=True,
        completely_shuffle=completely_shuffle,
        random_seed=12345,
    )

    iter_num = 10
    for i in range(iter_num):
        results = reader()
        datas = [r.numpy() for r in results]

        results_ = reader_()
        datas_ = [r.numpy() for r in results_]

        for (data, data_) in zip(datas, datas_):
            test_case.assertTrue(
                np.allclose(data, data_),
                f"{data}\n**** vs. ****\n{data_}\ndiff:\n{data - data_}",
            )


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class ParquetReaderTestCase(oneflow.unittest.TestCase):
    def test_compare_with_ofrecord(test_case):
        parquet_file = "/minio/sdb/dataset/criteo_kaggle/dlrm_parquet/train"
        schema = [
            {"col_id": 0, "shape": (1,), "dtype": flow.double},
            # {"col_id": 1, "shape": (13,), "dtype": flow.double},
            # {"col_id": 2, "shape": (26,), "dtype": flow.int32},
        ]
        batch_size = 100

        parquet_reader = make_parquet_reader(
            parquet_file, schema, batch_size=batch_size, shuffle=False
        )

        # ofrecord_file_dir = "/dataset/dlrm_ofrecord/"
        # ofrecord_reader = OFRecordReader(
        #     data_dir=ofrecord_file_dir,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     mode="train",
        # )

        iter_num = 10000
        for i in range(iter_num):
            results = parquet_reader()

            if i == 1:
                start = time.perf_counter()

            if i == iter_num - 1:
                datas = [r.numpy() for r in results]
                end = time.perf_counter()
                print(f"### escalped: {end -start}")

    @unittest.skipIf(True, "")
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
    def test_compare_consistent_with_ddp(test_case):
        parquet_dir = "/dataset/dlrm_parquet/train"
        schema = [
            {"col_id": 0, "shape": (), "dtype": flow.double},
            {"col_id": 1, "shape": (13,), "dtype": flow.double},
            {"col_id": 2, "shape": (26,), "dtype": flow.int32},
        ]
        batch_size = 32

        consistent_reader = make_parquet_reader(
            path=parquet_dir,
            schema=copy.deepcopy(schema),
            batch_size=batch_size * 2,
            shuffle=True,
            random_seed=12345,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0),
        )

        ddp_reader = make_parquet_reader(
            parquet_dir,
            schema,
            batch_size=batch_size,
            shuffle=True,
            random_seed=12345,
        )

        iter_num = 10
        for i in range(iter_num):
            results = consistent_reader()
            datas = [r.to_local().numpy() for r in results]

            results_ = ddp_reader()
            datas_ = [r.numpy() for r in results_]

            for j, (data, data_) in enumerate(zip(datas, datas_)):
                test_case.assertTrue(
                    np.allclose(data, data_),
                    f"## rank: {flow.env.get_rank()}, iter: {i}, out: {j} ##"
                    f"\n{data}\n**** vs. ****\n{data_}\ndiff:\n{data - data_}",
                )


if __name__ == "__main__":
    unittest.main()
