import oneflow as of
import torch
import numpy as np
import os

of.config.ctrl_port(19739)

def Save(name):
    def _save(x):
        path = "/tmp/compare_slice/"
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path + name, x.ndarray())
    return _save

def test_grad_2d(device_type):
    @of.function
    def slice_job(
        x=of.input_blob_def(
            (20, 20), dtype=of.float32, is_dynamic=True
        )
    ):
        of.config.train.primary_lr(0.0001)
        of.config.train.model_update_conf(dict(naive_conf={}))
        y = of.get_variable(
            name="y",
            shape=(1,),
            dtype=of.float32,
            initializer=of.constant_initializer(value=0.0),
        )
        with of.device_prior_placement(device_type, "0:0"):
            input = x + y
            output = of.slice_v2(input, [(3, -1, 2), (0, 8, 2)])

            of.watch(output, Save("of_out"))
            of.watch_diff(input, Save("of_in_diff"))

            of.losses.add_loss(output)

        return input, output

    # OneFlow
    check_point = of.train.CheckPoint()
    check_point.init()
    x = np.arange(100).reshape((10, 10)).astype(np.float32)
    of_input, of_out = slice_job(x).get()

    # PyTorch
    torch_input = torch.tensor(of_input, requires_grad=True)
    torch_out = torch_input[3:-1:2, 0:8:2]
    torch_out.sum().backward()

    np.allclose(np.load("/tmp/compare_slice/of_out.npy"), torch_out.detach().numpy())
    np.allclose(np.load("/tmp/compare_slice/of_in_diff.npy"), torch_input.grad.detach().numpy())

if __name__ == "__main__":
    test_grad_2d("gpu")
    # test_grad_2d("cpu")
