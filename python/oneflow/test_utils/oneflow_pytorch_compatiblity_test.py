import unittest

import numpy as np
import time
import argparse

import oneflow as flow
import torch
import oneflow.unittest

from models.oneflow_resnet import resnet50 as oneflow_resnet50
from models.pytorch_resnet import resnet50 as pytorch_resnet50

def test_train_loss_oneflow_pytorch(test_case, oneflow_model, pytorch_model, device):
    batch_size = 16
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)
    oneflow_model_loss = []
    pytorch_model_loss = []

    image = flow.tensor(image_nd)
    label = flow.tensor(label_nd)
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    image_gpu = image.to(device)
    label = label.to(device)
    oneflow_model.to(device)
    corss_entropy.to(device)

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

    print("pytorch traning loop avg time : {}".format((end_t - start_t) / bp_iters))
    print("forward avg time : {}".format(for_time / bp_iters))
    print("backward avg time : {}".format(bp_time / bp_iters))
    print("update parameters avg time : {}".format(update_time / bp_iters))
    print(oneflow_model_loss)
    print(pytorch_model_loss)

    test_case.assertTrue(np.allclose(oneflow_model_loss, pytorch_model_loss, 1e-03, 1e-03))


@flow.unittest.skip_unless_1n1d()
class TestApiCompatiblity(flow.unittest.TestCase):
    def test_resnet50_compatiblity(test_case):
        test_train_loss_oneflow_pytorch(test_case, oneflow_resnet50(), pytorch_resnet50(), "cuda")

if __name__ == "__main__":
    unittest.main()
