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


import oneflow as flow
import oneflow.unittest


ONEREC_URL = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/onerec_test/part-00000-713a0aee-1337-4686-b418-0ada6face4de-c000.onerec"
MD5 = "cc857a451cd796c12ff5d293177d1d24"


def md5(fname):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    result = hash_md5.hexdigest()
    print("md5", fname, result)
    return result


def download_file(out_path: str, url):
    import requests
    from tqdm import tqdm

    resp = requests.get(url=url, stream=True)
    MB = 1024 ** 2
    size = int(resp.headers["Content-Length"]) / MB
    print("File size: %.4f MB, downloading..." % size)
    with open(out_path, "wb") as f:
        for data in tqdm(
            iterable=resp.iter_content(MB), total=size, unit="m", desc=out_path
        ):
            f.write(data)
        print("Done!")


def ensure_dataset():
    import os
    import pathlib

    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "onerec_test"
    )
    file_path = pathlib.Path(data_dir) / ONEREC_URL.split("/")[-1]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_file_path = str(file_path.absolute())
    if file_path.exists():
        if MD5 != md5(absolute_file_path):
            file_path.unlink()
            download_file(absolute_file_path, ONEREC_URL)
    else:
        download_file(str(absolute_file_path), ONEREC_URL)
    assert MD5 == md5(absolute_file_path)
    return absolute_file_path


@flow.unittest.skip_unless_1n1d()
class TestOneRecOpsModule(flow.unittest.TestCase):
    def test_read_decode(test_case):
        files = [ensure_dataset()]
        onerec_reader = flow.nn.OneRecReader(
            files, batch_size=10, shuffle=True, shuffle_mode="batch"
        )
        readdata = onerec_reader()
        labels = flow.decode_onerec(
            readdata, key="labels", dtype=flow.int32, shape=(1,)
        )
        dense_fields = flow.decode_onerec(
            readdata, key="dense_fields", dtype=flow.float, shape=(13,)
        )
        test_case.assertTrue(labels.shape == (10, 1))
        test_case.assertTrue(dense_fields.shape == (10, 13))

    def test_global_one_rec(test_case):
        batch_size = 10
        files = [ensure_dataset()]
        onerec_reader = flow.nn.OneRecReader(
            files,
            batch_size=batch_size,
            shuffle=True,
            shuffle_mode="batch",
            placement=flow.placement("cpu", ranks=[0]),
            sbp=[flow.sbp.split(0)],
        )
        record_reader = onerec_reader()
        label = flow.decode_onerec(
            record_reader, key="labels", dtype=flow.int32, shape=(1,)
        )
        test_case.assertEqual(label.to_local().numpy().shape, (10, 1))
        dense_fields = flow.decode_onerec(
            record_reader, key="dense_fields", dtype=flow.float, shape=(13,)
        )
        test_case.assertEqual(dense_fields.to_local().numpy().shape, (10, 13))


if __name__ == "__main__":
    unittest.main()
