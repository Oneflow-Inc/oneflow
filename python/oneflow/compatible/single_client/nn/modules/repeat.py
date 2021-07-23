from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.python.nn.module import Module


class Repeat(Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        self.sizes = sizes

    def forward(self, input):
        repeat = self.sizes
        for repeat_v in repeat:
            assert repeat_v > 0
        input_shape = input.shape
        assert len(repeat) >= len(input_shape)
        in_reshape = []
        out_reshape = []
        expand_dim = []
        diff = len(repeat) - len(input_shape)
        for i in range(len(repeat) - 1, -1, -1):
            if i >= diff:
                if repeat[i] > 1:
                    if input_shape[i - diff] > 1:
                        in_reshape.insert(0, input_shape[i - diff])
                        in_reshape.insert(0, 1)
                        expand_dim.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, input_shape[i - diff] * repeat[i])
                    else:
                        in_reshape.insert(0, input_shape[i - diff])
                        expand_dim.insert(0, repeat[i])
                        out_reshape.insert(0, repeat[i])
                else:
                    in_reshape.insert(0, input_shape[i - diff])
                    expand_dim.insert(0, input_shape[i - diff])
                    out_reshape.insert(0, input_shape[i - diff])
            else:
                expand_dim.insert(0, repeat[i])
                out_reshape.insert(0, repeat[i])
        new_tensor = flow.experimental.reshape(input, in_reshape)
        tmp_tensor = new_tensor.expand(*expand_dim)
        out = flow.experimental.reshape(tmp_tensor, out_reshape)
        return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
