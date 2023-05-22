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
import importlib.util
import unittest

import numpy as np
import time
import tempfile
import argparse

import oneflow as flow
import torch
import oneflow.unittest
import shutil
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def import_file(source):
    with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
        f.write(source)
        f.flush()
        spec = importlib.util.spec_from_file_location("mod", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def get_loss(
    image_nd,
    label_nd,
    model_path: str,
    module_name: str,
    test_pytorch: bool = True,
    device: str = "cuda",
    tmpfilename: str = "/tmp/oneflow_tmp_file",
):
    model_loss = []
    learning_rate = 0.01
    mom = 0.9
    bp_iters = 100

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    if test_pytorch == True:
        image = flow.tensor(image_nd)
        label = flow.tensor(label_nd)
        corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

        with open(model_path) as f:
            buf = f.read()
            lines = buf.split("\n")
            buf = "\n".join(lines)
            python_module = import_file(buf)

        Net = getattr(python_module, module_name)
        pytorch_model = Net()

        w = pytorch_model.state_dict()
        new_parameters = dict()
        for k, v in w.items():
            if "num_batches_tracked" not in k:
                new_parameters[k] = flow.tensor(w[k].detach().numpy())

        flow.save(new_parameters, tmpfilename)

        pytorch_model.to(device)
        torch_sgd = torch.optim.SGD(
            pytorch_model.parameters(), lr=learning_rate, momentum=mom
        )

        image = torch.tensor(image_nd)
        image_gpu = image.to(device)
        corss_entropy = torch.nn.CrossEntropyLoss()
        corss_entropy.to(device)
        label = torch.tensor(label_nd, dtype=torch.long).to(device)

        print("start pytorch training loop....")
        start_t = time.time()
        for i in range(bp_iters):
            s_t = time.time()
            logits = pytorch_model(image_gpu)
            loss = corss_entropy(logits, label)
            for_time += time.time() - s_t

            s_t = time.time()
            loss.backward()
            bp_time += time.time() - s_t

            model_loss.append(loss.detach().cpu().numpy())

            s_t = time.time()
            torch_sgd.step()
            torch_sgd.zero_grad()
            update_time += time.time() - s_t

        end_t = time.time()

        if verbose:
            print(
                "pytorch traning loop avg time : {}".format(
                    (end_t - start_t) / bp_iters
                )
            )
            print("forward avg time : {}".format(for_time / bp_iters))
            print("backward avg time : {}".format(bp_time / bp_iters))
            print("update parameters avg time : {}".format(update_time / bp_iters))
    else:
        with open(model_path) as f:
            buf = f.read()

            lines = buf.split("\n")
            for i, line in enumerate(lines):
                if (
                    i > 15 and "import" not in line and len(line.strip()) != 0
                ):  # 15 means license
                    break
            lines = (
                lines[:i]
                + [
                    "import oneflow as torch",
                    "import oneflow.nn as nn",
                    "import oneflow.nn.init as init",
                    "import oneflow.nn.functional as F",
                    "from oneflow import Tensor",
                    "from oneflow.nn import Parameter",
                    "import math",
                    "from flowvision.layers import *",
                ]
                + lines[i:]
            )
            buf = "\n".join(lines)

            python_module = import_file(buf)

        Net = getattr(python_module, module_name)
        oneflow_model = Net()

        image = flow.tensor(image_nd)
        label = flow.tensor(label_nd)
        corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

        image_gpu = image.to(device)
        label = label.to(device)
        oneflow_model.to(device)
        corss_entropy.to(device)

        params = flow.load(tmpfilename)
        oneflow_model.load_state_dict(params)

        of_sgd = flow.optim.SGD(
            oneflow_model.parameters(), lr=learning_rate, momentum=mom
        )

        print("start oneflow training loop....")
        start_t = time.time()
        for i in range(bp_iters):
            s_t = time.time()
            logits = oneflow_model(image_gpu)
            loss = corss_entropy(logits, label)
            for_time += time.time() - s_t

            s_t = time.time()
            loss.backward()
            bp_time += time.time() - s_t

            model_loss.append(loss.numpy())

            s_t = time.time()
            of_sgd.step()
            of_sgd.zero_grad()
            update_time += time.time() - s_t

        end_t = time.time()

        if verbose:
            print(
                "oneflow traning loop avg time : {}".format(
                    (end_t - start_t) / bp_iters
                )
            )
            print("forward avg time : {}".format(for_time / bp_iters))
            print("backward avg time : {}".format(bp_time / bp_iters))
            print("update parameters avg time : {}".format(update_time / bp_iters))

    return model_loss


def do_test_train_loss_oneflow_pytorch(
    test_case,
    model_path: str,
    module_name: str,
    device: str = "cuda",
    batch_size: int = 16,
    img_size: int = 224,
):
    image_nd = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)
    oneflow_model_loss = []
    pytorch_model_loss = []

    with tempfile.NamedTemporaryFile() as f:
        pytorch_model_loss = get_loss(
            image_nd, label_nd, model_path, module_name, True, device, f.name
        )
        oneflow_model_loss = get_loss(
            image_nd, label_nd, model_path, module_name, False, device, f.name
        )

    if verbose:
        indes = [i for i in range(len(oneflow_model_loss))]

        plt.plot(indes, oneflow_model_loss, label="oneflow")
        plt.plot(indes, pytorch_model_loss, label="pytorch")

        plt.xlabel("iter - axis")
        # Set the y axis label of the current axis.
        plt.ylabel("loss - axis")
        # Set a title of the current axes.
        plt.title("compare ")
        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.savefig("./loss_compare.png")
        plt.show()

    test_case.assertTrue(
        np.allclose(cos_sim(oneflow_model_loss, pytorch_model_loss), 1.0, 1e-1, 1e-1)
    )
