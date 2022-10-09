import numpy as np
import oneflow
import torch
from ctypes import addressof
import pdb

def test_tensor_update():
    init_x = np.random.uniform(size=10).astype(np.float32)

    def torch_con_update():
        x = torch.tensor(init_x, requires_grad=True, dtype=torch.float32)
        y = torch.zeros(x.numel())
        z = torch.zeros(x.numel())
        y[0:len(y)] = x.data.view(-1)
        x.data = y[:]
        x.grad = z[:]
        y.grad = z
        def chk():
            print(id(x.grad), id(z.data), id(y.grad))
            print(x.grad)
            print(z.data)
            print(y.grad)
            print()
        sgd = torch.optim.SGD([y], lr=0.1, weight_decay=0.9)
        loss = torch.sum(x)
        chk()
        loss.backward()
        chk()
        sgd.step()
        print('----------------------')
        return x

    def oneflow_con_update():
        x = oneflow.tensor(init_x, requires_grad=True, dtype=oneflow.float32)
        y = oneflow.zeros(x.numel())
        z = oneflow.zeros(x.numel())
        y[0:len(y)] = x.data.view(-1)
        x.data = y[:]
        x.grad = z[:]
        y.grad = x.grad
        def chk():
            print(id(x.grad), id(z.data), id(y.grad))
            y.grad += 1
            print(x.grad)
            print(z.data)
            print(y.grad)
            print()
        sgd = oneflow.optim.SGD([y], lr=0.1, weight_decay=0.9)
        loss = oneflow.sum(x)
        chk()
        loss.backward()
        chk()
        sgd.step()
        return x

    def torch_update():
        x = torch.tensor(init_x, requires_grad=True, dtype=torch.float32)
        sgd = torch.optim.SGD([x], lr=0.1, weight_decay=0.9)
        loss = torch.sum(x)
        loss.backward()
        sgd.step()
        return x

    def oneflow_update():
        x = oneflow.tensor(init_x, requires_grad=True, dtype=oneflow.float32)
        loss = oneflow.sum(x)
        loss.backward()
        sgd = oneflow.optim.SGD([x], lr=0.1, weight_decay=0.9)
        sgd.step()
        return x

    a, b = torch_update().detach().numpy(), oneflow_update().numpy()
    c, d = torch_con_update().detach().numpy(), oneflow_con_update().numpy()
    assert(np.allclose(a, b, atol=1e-4))
    assert(np.allclose(a, c, atol=1e-4))
    assert(np.allclose(a, d, atol=1e-4))

if __name__ == "__main__":
    test_tensor_update()