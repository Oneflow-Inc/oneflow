from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.python.nn.module import Module
from oneflow.compatible.single_client.python.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from oneflow.compatible.single_client.python.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_any_t
from typing import Optional, List, Tuple, Sequence, Union
import random
import sys
import traceback

def mirrored_gen_random_seed(seed=None):
    if seed is None:
        seed = -1
        has_seed = False
    else:
        has_seed = True
    return (seed, has_seed)

class TensorBufferToListOfTensors(Module):

    def __init__(self, out_shapes, out_dtypes, out_num: int=1, dynamic_out: bool=False):
        super().__init__()
        self._op = flow.builtin_op('tensor_buffer_to_list_of_tensors_v2').Input('in').Output('out', out_num).Attr('out_shapes', out_shapes).Attr('out_dtypes', out_dtypes).Attr('dynamic_out', dynamic_out).Build()

    def forward(self, input):
        return self._op(input)