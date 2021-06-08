import oneflow as flow
from typing import List, Tuple
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
import oneflow.python.nn as nn


@oneflow_export("stack")
@register_tensor_op("stack")
@experimental_api
def Stack(inputs, dim=0):
    _input_shape = inputs[0].shape
    _max_dim = len(_input_shape)

    assert isinstance(inputs, (List, Tuple))
    _input_shape = inputs[0].shape
    _max_dim = len(_input_shape)

    # The axis must be in range [-(_max_dim +1), _max_dim]
    if dim < 0:
        dim = dim + _max_dim + 1
    assert (dim >= 0) and (dim <= _max_dim)
    _input_list_length = len(inputs)
    for i in range(_input_list_length):
        _current_shape = inputs[i].shape
        assert (
                _input_shape == _current_shape
        ), "Each tensor should have the same shape ! Found a tensor instance shape is: {}".format(
            _current_shape
        )
        inputs[i] = flow.experimental.unsqueeze(inputs[i], dim=dim)
    return flow.experimental.cat(inputs, dim=dim)
