import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
def compare_with_numpy_sgd(
    test_case,
    device,
    x_shape,
    scale,
    learning_rate,
    train_iters,
    momentum,
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

        def forward(self, mask):
            return self.para0 * mask


    simp_module = CustomModule()
    simp_module.to("cuda")
    simp_module.train()
    # for params in simp_module.parameters(): 
    print(simp_module.para0)
    adam0 = flow.optim.SGD(
            [
                {
                    "params": simp_module.parameters(),
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "scale": scale,
                }
            ]
        )

    class CustomAdamGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = simp_module
            self.add_optimizer("adam", adam0)

        def build(self, mask_tensor):
            loss = flow.sum(self.m(mask_tensor))
            loss.backward()
            print("Graph params print: ", self.m.origin.para0)
            return loss
            # return self.m.origin.para0
            # print("return type is: ", type(self.m.para0))
            # print("return type is: ", type(flow.Tensor(self.m.origin.para0)))
            # return self.m.para0
            # return flow.Tensor(self.m.origin.para0)

    for i in range(train_iters):
        adam_graph = CustomAdamGraph()
        mask_tensor = flow.Tensor(
                random_grad_seq[i], requires_grad=False, device=flow.device(device)
            )
        adam_x = adam_graph(mask_tensor)

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def train_by_numpy():
            x = init_value
            vt = np.zeros_like(x)

        def np_train_one_iter(grad):
            grad = grad * scale + weight_decay * x
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)


        print("Numpy input is: ", x)
        for i in range(train_iters):
            print("train iter: {} :".format(i))
            print("===== the grad is: {} =====".format(random_grad_seq[i]))
            (x, vt) = np_train_one_iter(random_grad_seq[i])
            print("the params is: {}".format(x))
        return x

    oneflow_res = adam_x.numpy()
    numpy_res = train_by_numpy()
    print("Oneflow res is: ", oneflow_res)
    print("Numpy res is: ", numpy_res)

    # test_case.assertTrue(
    #     np.allclose(oneflow_res.flatten(), numpy_res.flatten(), rtol=0.001, atol=0.001)
    # )


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
        compare_with_numpy_sgd(test_case, 
                                device="cuda", 
                                x_shape=(1, ), 
                                scale=1.0, 
                                # learning_rate=0.00001, 
                                learning_rate=1, 
                                momentum=0.9, 
                                train_iters=10, 
                                weight_decay=0.0, 
                                eps=1e-8)


if __name__ == "__main__":
    unittest.main()