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

ONEREC_URL = (
    "https://oneflow-public.oss-cn-beijing.aliyuncs.com/sync_bn_test_datas.tar.gz"
)
MD5 = "537ff00fb47be8be90df75f47a883b76"


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


def ensure_datas():
    import os
    import pathlib

    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "sync_bn"
    )
    file_path = pathlib.Path(data_dir) / ONEREC_URL.split("/")[-1]
    absolute_file_path = str(file_path.absolute())

    if flow.env.get_rank() == 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            if MD5 != md5(absolute_file_path):
                file_path.unlink()
                download_file(absolute_file_path, ONEREC_URL)
        else:
            download_file(str(absolute_file_path), ONEREC_URL)
        assert MD5 == md5(absolute_file_path)

        import tarfile

        my_tar = tarfile.open(str(absolute_file_path))
        my_tar.extractall(data_dir)  # specify which folder to extract to
        my_tar.close()

    flow.comm.barrier()

    return os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"),
        "sync_bn",
        "sync_bn_test_datas",
    )
