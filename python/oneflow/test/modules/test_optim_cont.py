"""Test contiguous parameter functions."""
from copy import deepcopy
import os
import pdb
import oneflow
import numpy as np
from oneflow import nn
from oneflow.nn.optimizer.contiguous_params import ContiguousParams

def test_equal_optimizer_update(device):
    if device == "cuda" and not oneflow.cuda.is_available():
        print("No GPU available, skipping GPU test.")
        return
    """Verify that the parameters are the same after a few updates."""
    xx, yy = [], []
    for i in range(5):
        xx.append(oneflow.randn(1, 8).to(device))
        print(xx[i].shape)
        yy.append(oneflow.randn(1, 8).to(device))
    ce = nn.CrossEntropyLoss()

    model_ref = nn.Sequential(*[nn.Linear(8, 8) for i in range(10)])
    model_ref = model_ref.to(device)
    optimizer = oneflow.optim.SGD(model_ref.parameters(), lr=1e-3)
    
    model_c = deepcopy(model_ref)
    parameters_c = ContiguousParams(model_c.parameters())
    optimizer_c = oneflow.optim.SGD(parameters_c.contiguous(), lr=1e-3)

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for i in range(5):
            print(i)
            x = oneflow.tensor(xx[i], device = xx[i].device)
            print(x.shape)
            y = oneflow.tensor(yy[i], device = yy[i].device)
            loss = ce(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Verify that the model/optimizer did not modify the data or grad handle.
    #parameters_c.assert_buffer_is_valid()

    # Verify that both models applied the same parameter updates.
    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        assert np.allclose(p1.data.numpy(), p2.data.numpy(), atol=1e-06)

test_equal_optimizer_update('cuda')
