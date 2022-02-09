"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import oneflow as flow
from oneflow.framework.tensor import Tensor, register_tensor_op


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def index_select_op(input, dim, index):
    assert len(index.shape) == 1, "Dimensions of index should be 1-D"
    assert (
        dim < len(input.shape) and dim >= 0
    ), "Value of dim is out of range(dim should be in the range of [0, input dimensions) )"
    assert _input_args_is_int(
        index.tolist()
    ), "input index parameter is not legal!(index should be an 1-D int tensor)"
    index_rshp = list(input.shape)

    for index_i in index:
        assert (
            index_i < index_rshp[dim]
        ), "index is out of range(index shuold be lower than the dim-th dimension of input)"

    index_rshp[dim] = 1
    index_gather = index[0].expand(*index_rshp)
    if index.size()[0] > 1:
        for index_i in index[1:]:
            x = index_i.expand(*index_rshp)
            index_gather = flow.cat((index_gather, x), dim)

    return flow.gather(input, dim, index_gather)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)
