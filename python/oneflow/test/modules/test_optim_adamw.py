import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow
from oneflow.nn.parameter import Parameter


def compare_with_numpy_adamw(
    test_case, device, x_shape, scale, learning_rate, train_iters, weight_decay
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        adam = flow.optim.AdamW(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "scale": scale,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.Tensor(
                grad, requires_grad=False, device=flow.device(device)
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            adam.step()
            adam.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)
        st = np.zeros_like(x)
        beta1 = 0.9
        beta2 = 0.999

        def train_one_iter(grad):
            grad = grad * scale
            v = beta1 * vt + (1 - beta1) * grad
            s = beta2 * st + (1 - beta2) * grad * grad
            g = (
                learning_rate / (np.sqrt(s) + 1e-08) * v
                + learning_rate * weight_decay * x
            )
            param = x - g
            return (param, v, s)

        for i in range(train_iters):
            (x, vt, st) = train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()
    test_case.assertTrue(
        np.allclose(
            oneflow_res.flatten(), numpy_res.flatten(), rtol=0.0001, atol=0.0001
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestAdamW(flow.unittest.TestCase):
    def test_adamw(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["scale"] = [1.0, 0.9]
        arg_dict["learning_rate"] = [1]
        arg_dict["train_iters"] = [10]
        arg_dict["weight_decay"] = [0.001, 0.0]
        for arg in GenArgList(arg_dict):
            compare_with_numpy_adamw(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
