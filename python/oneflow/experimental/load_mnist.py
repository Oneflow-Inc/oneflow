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
import hashlib
import os

import numpy as np
import requests
from tqdm import tqdm


def get_sha256hash(file_path, Bytes=1024):
    sha256hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(Bytes)
            if data:
                sha256hash.update(data)
            else:
                break
    ret = sha256hash.hexdigest()
    return ret


def download_mnist_file(out_path, url):
    resp = requests.get(url=url, stream=True)
    size = int(resp.headers["Content-Length"]) / 1024
    print("File size: %.4f kb, downloading..." % size)
    with open(out_path, "wb") as f:
        for data in tqdm(
            iterable=resp.iter_content(1024), total=size, unit="k", desc=out_path
        ):
            f.write(data)
        print("Done!")


def get_mnist_file(sha256, url, out_dir):
    path = os.path.join(out_dir, "mnist.npz")
    if not os.path.isfile(path):
        download_mnist_file(path, url)
    print("File mnist.npz already exist, path:", path)
    if not get_sha256hash(path) == sha256:
        cheksum_fail = "sha256 verification failed, remove {0} and try again".format(
            path
        )
        raise Exception(cheksum_fail)
    return path


def load_mnist(
    train_batch_size=100,
    test_batch_size=100,
    data_format="NCHW",
    url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist.npz",
    hash_check="63d4344077849053dc3036b247fa012b2b381de53fd055a66b539dffd76cf08e",
    out_dir=".",
):
    """Load mnist dataset, return images and labels,
            if  dataset doesn't exist, then download it to directory that out_dir specified

    Args:
        train_batch_size (int, optional): batch size for train. Defaults to 100.
        test_batch_size (int, optional): batch size for test or evaluate. Defaults to 100.
        data_format (str, optional): data format. Defaults to "NCHW".
        url (str, optional): url to get mnist.npz. Defaults to "https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist.npz".
        hash_check (str, optional): file hash value. Defaults to "63d4344077849053dc3036b247fa012b2b381de53fd055a66b539dffd76cf08e".
        out_dir (str, optional): dir to save downloaded file. Defaults to "./".

    Returns:
        (train_images, train_labels), (test_images, test_labels)
    """
    path = get_mnist_file(hash_check, url, out_dir)
    with np.load(path, allow_pickle=True) as f:
        (x_train, y_train) = (f["x_train"], f["y_train"])
        (x_test, y_test) = (f["x_test"], f["y_test"])

    def normalize(x, y, batch_size):
        x = x.astype(np.float32) / 255.0
        y = y.astype(np.int32)
        if data_format == "NCHW":
            images = x.reshape((-1, batch_size, 1, x.shape[1], x.shape[2]))
        else:
            images = x.reshape((-1, batch_size, x.shape[1], x.shape[2], 1))
        labels = y.reshape((-1, batch_size))
        return (images, labels)

    (train_images, train_labels) = normalize(x_train, y_train, train_batch_size)
    (test_images, test_labels) = normalize(x_test, y_test, test_batch_size)
    return ((train_images, train_labels), (test_images, test_labels))
