from typing import Union
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.framework.tensor import Tensor, register_tensor_op

class Tile(Module):

    def __init__(self, reps: tuple) -> None:
        super().__init__()
        self.reps = reps

    def forward(self, input: Tensor) -> Tensor:
        reps = self.reps
        for s in self.reps:
            assert s > 0
        input_shape = input.shape
        diff = len(input_shape) - len(reps)
        if diff > 0:
            shape = [1 for _ in range(diff)]
            shape.extend([i for i in reps])
            reps = tuple(shape)
        return input.repeat(reps)
if __name__ == '__main__':
    import doctest
    doctest.testmod(raise_on_error=True)