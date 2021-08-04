import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
def compare_with_numpy_adam(
    test_case,
    device,
    x_shape,
    scale,
    learning_rate,
    train_iters,
    betas,
    weight_decay,
    eps,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.para0 = flow.nn.Parameter(flow.Tensor(init_value, device=flow.device(device)))

        def forward(self, grad):
            return self.para0 * grad


    simp_module = CustomModule()
    simp_module.to("cuda")


    adam0 = flow.optim.Adam(
            [
                {
                    "params": simp_module.parameters(),
                    "lr": learning_rate,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "scale": scale,
                }
            ]
        )

    class CustomAdamGraph(flow.nn.Graph):
        def __init__(self, grad):
            super().__init__()
            self.m = simp_module
            self.grad_tensor = flow.Tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            self.add_optimizer("adam", adam0)

        def build(self):
            loss = flow.sum(self.m(self.grad_tensor))
            loss.backward()
            return loss


    for i in range(train_iters):
        adam_graph = CustomAdamGraph(random_grad_seq[i])
        adam_x = adam_graph()

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        beta1 = betas[0]
        beta2 = betas[1]

        def np_rain_one_iter(grad):
            grad = grad * scale + weight_decay * x
            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            param = x - learning_rate * (v / (np.sqrt(s) + eps))
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = np_train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = adam_x.numpy()
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=0.001, atol=0.001)
    )


@flow.unittest.skip_unless_1n1d()
class TestAdam(flow.unittest.TestCase):
    def test_adam1(test_case):
        # arg_dict = OrderedDict()
        # arg_dict["device"] = ["cpu", "cuda"]
        # arg_dict["x_shape"] = [(10,)]
        # arg_dict["scale"] = [1.0, 0.8]
        # arg_dict["learning_rate"] = [1]
        # arg_dict["train_iters"] = [10]
        # arg_dict["betas"] = [(0.99, 0.9), (0.8, 0.7)]
        # arg_dict["weight_decay"] = [0.0, 0.1]
        # arg_dict["eps"] = [1e-08, 1e-07]
        compare_with_numpy_adam(test_case, 
                                device="cuda", 
                                x_shape=(10, ), 
                                scale=1.0, 
                                learning_rate=1, 
                                train_iters=10, 
                                betas=(0.99, 0.9), 
                                weight_decay=0.0, 
                                eps=1e-8)


if __name__ == "__main__":
    unittest.main()