from copy import deepcopy
import oneflow
from oneflow.nn.optimizer.contiguous_params import ContiguousParams
import oneflow.optim as optim
import oneflow.nn as nn
import flowvision

import unittest
from collections import OrderedDict
import os
import numpy as np
from oneflow.test_utils.test_util import GenArgDict

import oneflow as flow
from oneflow.nn.parameter import Parameter

def test_sgd_train(test_case):
    train_iter = 10
    learning_rate, weight_decay = 0.1, 0.9

    model_ref = flowvision.models.resnet.resnet101(pretrained=True).to("cuda")
    criterion = nn.CrossEntropyLoss()

    # optimizers
    optimizer_ref = optim.SGD(
        [
            {
                "params": model_ref.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    )

    model_c = deepcopy(model_ref)
    param_c = ContiguousParams(model_c.parameters())
    optim_c = flow.optim.SGD(
        [
            {
                "params": param_c.contiguous(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    )

    x, y = [], []
    for _ in range(train_iter):
        x.append(oneflow.rand(128, 3, 100, 100, device="cuda" , requires_grad=True))
        y.append(oneflow.randint(0, 5, (128,), dtype=oneflow.long, device='cuda'))
    for model, optimizer in zip([model_ref, model_c], [optimizer_ref, optim_c]):
        for i in range(train_iter):
            xx = oneflow.tensor(x[i], device='cuda')
            yy = oneflow.tensor(y[i], device='cuda')
            print(list(model.parameters())[0][0][0][0])
            optimizer.zero_grad()
            outputs = model(xx)
            loss = criterion(outputs, yy)
            loss.backward()
            optimizer.step()
        print()
    
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert np.allclose(p1.numpy(), p2.numpy(), atol=1e-06)

@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_multi_tensor_sgd_update(test_case):
        test_sgd_train(test_case)

if __name__ == "__main__":
    unittest.main()