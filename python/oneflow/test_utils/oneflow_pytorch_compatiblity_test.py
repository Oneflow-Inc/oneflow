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
mpl.use('Agg')
import matplotlib.pyplot as plt

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def import_file(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_train_loss_oneflow_pytorch(test_case, model_path: str, module_name: str, device: str="cuda"):
    batch_size = 16
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)
    oneflow_model_loss = []
    pytorch_model_loss = []

    verbose = os.getenv("ONEFLOW_TEST_VERBOSE") is not None
    image = flow.tensor(image_nd)
    label = flow.tensor(label_nd)
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    python_module = import_file(model_path)
    Net = getattr(python_module, module_name)
    pytorch_model = Net()

    with open(model_path) as f:
        buf = f.read()

        lines = buf.split("\n")
        for i, line in enumerate(lines):
            if "import" not in line and len(line.strip()) != 0:
                break
        lines = (
            lines[:i]
            + [
                "import torch as flow",
                "import torch.nn as nn",
                "from torch import Tensor",
                "from torch.nn import Parameter",
            ]
            + lines[i:]
        )
        buf = "\n".join(lines)
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(buf)
            torch_python_module = import_file(f.name)
            print(torch_python_module)

    Net = getattr(torch_python_module, module_name)
    oneflow_model = Net()

    image_gpu = image.to(device)
    label = label.to(device)
    oneflow_model.to(device)
    corss_entropy.to(device)

    w = pytorch_model.state_dict()
    new_parameters = dict()
    for k, v in w.items():
        if "num_batches_tracked" not in k:
            new_parameters[k] = flow.tensor(w[k].detach().numpy())

    flow.save(new_parameters, "/dataset/imagenet/compatiblity_models")
    params = flow.load("/dataset/imagenet/compatiblity_models")
    oneflow_model.load_state_dict(params)

    learning_rate = 0.01
    mom = 0.9
    of_sgd = flow.optim.SGD(oneflow_model.parameters(), lr=learning_rate, momentum=mom)

    bp_iters = 100
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

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

        oneflow_model_loss.append(loss.numpy())

        s_t = time.time()
        of_sgd.step()
        of_sgd.zero_grad()
        update_time += time.time() - s_t

    end_t = time.time()

    if verbose:
        print("oneflow traning loop avg time : {}".format((end_t - start_t) / bp_iters))
        print("forward avg time : {}".format(for_time / bp_iters))
        print("backward avg time : {}".format(bp_time / bp_iters))
        print("update parameters avg time : {}".format(update_time / bp_iters))

    #####################################################################################################

    # set for eval mode
    # pytorch_model.eval()
    pytorch_model.to(device)
    torch_sgd = torch.optim.SGD(
        pytorch_model.parameters(), lr=learning_rate, momentum=mom
    )

    image = torch.tensor(image_nd)
    image_gpu = image.to(device)
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to(device)
    label = torch.tensor(label_nd, dtype=torch.long).to(device)

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

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

        pytorch_model_loss.append(loss.detach().cpu().numpy())

        s_t = time.time()
        torch_sgd.step()
        torch_sgd.zero_grad()
        update_time += time.time() - s_t

    end_t = time.time()

    if verbose:
        print("pytorch traning loop avg time : {}".format((end_t - start_t) / bp_iters))
        print("forward avg time : {}".format(for_time / bp_iters))
        print("backward avg time : {}".format(bp_time / bp_iters))
        print("update parameters avg time : {}".format(update_time / bp_iters))
        for i in range(100):
            print(f'oneflow_loss:{oneflow_model_loss[i]}, pytorch_loss:{pytorch_model_loss[i]}')

    shutil.rmtree("/dataset/imagenet/compatiblity_models")

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
        plt.savefig('./loss_compare.png')
        plt.show()

    test_case.assertTrue(
        np.allclose(cos_sim(oneflow_model_loss, pytorch_model_loss), 1.0, 1e-1, 1e-1)
    )


@flow.unittest.skip_unless_1n1d()
class TestApiCompatiblity(flow.unittest.TestCase):
    def test_alexnet_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(
            test_case, "pytorch_alexnet.py", "alexnet", "cuda"
        )
    
    # def test_resnet50_compatiblity(test_case):
    #     test_train_loss_oneflow_pytorch(
    #         test_case, "cuda"
    #     )


if __name__ == "__main__":
    unittest.main()
