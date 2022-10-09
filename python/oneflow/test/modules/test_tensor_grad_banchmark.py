import pdb
import oneflow, torch
import numpy as np

def test_case():
    def torch_case():
        a = torch.ones(4, requires_grad=True)
        b = torch.ones(4)
        a.grad = b[:]
        loss = torch.sum(a)
        loss.backward()
        print(id(a.grad))
        print(id(a.grad.data))
        print(id(b))
        print(id(b.data))
        assert(np.allclose(a.grad.numpy(), b.numpy(), atol=1e-4))

    def oneflow_case():
        a = oneflow.ones(4, requires_grad=True)
        c = oneflow.ones(4)
        a.grad = c.data[:]
        a._is_grad_acc_inplace = True
        loss = oneflow.sum(a)
        loss.backward()
        print(id(a.grad))
        print(id(a.grad.data))
        print(id(c))
        print(id(c.data))
        assert(np.allclose(a.grad.numpy(), c.numpy(), atol=1e-4))
        
    torch_case()
    oneflow_case()

if __name__ == '__main__':
    test_case()
