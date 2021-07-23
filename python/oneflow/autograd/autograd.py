from typing import Union, Sequence, Tuple
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple
from oneflow._oneflow_internal import TensorTuple
from oneflow._oneflow_internal.autograd import grad as grad_api
from oneflow._oneflow_internal.autograd import backward as backward_api


def grad(
    outputs: Union[Tensor, Sequence[Tensor]],
    inputs: Union[Tensor, Sequence[Tensor]],
    out_grads: Union[Tensor, Sequence[Tensor], None] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor]:
    in_grads = grad_api(
        convert_to_tensor_tuple(outputs),
        convert_to_tensor_tuple(inputs),
        convert_to_tensor_tuple(out_grads),
        retain_graph,
        create_graph,
    )
    return tuple([Tensor(x) for x in in_grads])


def backward(
    outputs: Union[Tensor, Sequence[Tensor]],
    out_grads: Union[Tensor, Sequence[Tensor], None],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None:
    backward_api(
        convert_to_tensor_tuple(outputs),
        convert_to_tensor_tuple(out_grads),
        retain_graph,
        create_graph,
    )
