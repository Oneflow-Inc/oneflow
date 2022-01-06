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

import oneflow as flow
import oneflow.unittest


def make_parquet_reader(
    path,
    schema,
    batch_size=256,
    shuffle=True,
    random_seed=12345,
    prefetch_buffer_size=None,
    use_mmap=True,
    device=None,
    placement=None,
    sbp=None,
):
    return flow.nn.ParquetReader(
        path,
        schema,
        batch_size,
        shuffle,
        random_seed,
        prefetch_buffer_size,
        use_mmap,
        device,
        placement,
        sbp,
    )


def get_wdl_path():
    return "/dataset/wdl_parquet/train/"


def get_wdl_path_2():
    return "/dataset/dlrm_parquet/train/"


def get_wdl_schema():
    return [
        # {"col_id": 0, "shape": (26,), "dtype": flow.int32},
        {"col_id": 1, "shape": (13,), "dtype": flow.float32},
        # {"col_id": 2, "shape": (), "dtype": flow.int32},
        {"col_name": "labels", "shape": (), "dtype": flow.int32},
        # {"col_id": 3, "shape": (2,), "dtype": flow.int32},
    ]


def get_wdl_schema_2():
    return [
        {"col_name": "labels", "shape": (), "dtype": flow.double},
        {"col_id": 1, "shape": (13,), "dtype": flow.double},
        {"col_id": 2, "shape": (26,), "dtype": flow.int32},
    ]


class MyGraph(flow.nn.Graph):
    def __init__(self, reader):
        super().__init__()
        self.reader_ = reader

    def build(self):
        return self.reader_()


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class ParquetReaderTestCase(oneflow.unittest.TestCase):
    def test_wdl(test_case):
        reader = make_parquet_reader(get_wdl_path(), get_wdl_schema())
        columns = reader()
        for col in columns:
            print(col)

    def test_wdl_2(test_case):
        reader = make_parquet_reader(get_wdl_path_2(), get_wdl_schema_2())
        columns = reader()
        for col in columns:
            print(col)

    def test_imagenet(test_case):
        pass


@unittest.skipIf(
    os.getenv("ONEFLOW_TEST_GITHUB_HOSTED"),
    "/dataset not available on GitHub hosted servers",
)
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class ParquetReaderTestCaseDistributedTestCase(oneflow.unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
