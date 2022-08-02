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
from typing import Sequence, Tuple, Union

from oneflow._oneflow_internal import TensorTuple
from oneflow._oneflow_internal.autograd import backward as backward_api
from oneflow._oneflow_internal.autograd import grad as grad_api
from oneflow.framework.tensor import Tensor
from oneflow.framework.tensor_tuple_util import convert_to_tensor_tuple


def grad(
    outputs: Union[Tensor, Sequence[Tensor]],
    inputs: Union[Tensor, Sequence[Tensor]],
    grad_outputs: Union[Tensor, Sequence[Tensor], None] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor]:
    r"""
    Computes and returns the sum of gradients of outputs with respect to the inputs.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.autograd.grad.html.

    The graph is differentiated using the chain rule. ``grad_outputs`` should be a sequence of
    length matching ``outputs``, containing the "vector" in the Jacobian-vector product.
    (``None`` is an acceptable value for that tensor don't require gradient.)

    Args:
        outputs (Sequence[Tensor]): Tensors of which the derivative will be computed.
        inputs (Sequence[Tensor]): Inputs w.r.t. which the derivative will be returned(and not
            accumulated into ``.grad``).
        grad_outputs (Sequence[Tensor], optional): The "vector" in the Jacobian-vector product.
            Usually gradients w.r.t. each output. None values can be specified for scalar Tensors
            or ones that don't require grad. Defaults to None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be
            reset after backward is complete. Defaults to ``False``. Note that in nearly all cases
            setting this option to ``True`` is not needed and often can be worked around in a much
            more efficient way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Defaults to ``False``.

    Returns:
        Tuple(Tensor): A tuple of tensors containing the gradients for each ``inputs``.
    """
    in_grads = grad_api(
        convert_to_tensor_tuple(outputs),
        convert_to_tensor_tuple(inputs),
        convert_to_tensor_tuple(grad_outputs),
        retain_graph,
        create_graph,
    )
    return tuple([Tensor(x) for x in in_grads])


def backward(
    tensors: Union[Tensor, Sequence[Tensor]],
    grad_tensors: Union[Tensor, Sequence[Tensor], None],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> None:
    r"""
    Computes the sum of gradients of given tensors with respect to graph leaves.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.autograd.backward.html.

    The graph is differentiated using the chain rule. If any of ``tensors`` are non-scalar (i.e.
    their data has more than one element) and require gradient, then the Jacobian-vector product
    would be computed, in this case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector" in the Jacobian-vector
    product, usually the gradient of the differentiated function w.r.t. corresponding tensors.
    (``None`` is an acceptable value for all tensors that don't need gradient.)

    This function accumulates gradients in the leaves - you might need to zero ``.grad`` attributes
    or set them to ``None`` before calling it.

    Note:
        Using this method with ``create_graph=True`` will create a reference cycle between the
        parameter and its gradient which can cause a memory leak. We recommend using
        ``autograd.grad`` when creating the graph to avoid this. If you have to use this function,
        make sure to reset the ``.grad`` fields of your parameters to ``None`` after use to break
        the cycle and avoid the leak.

    Args:
        tensors (Tensor or Sequence[Tensor]): Tensors of which the derivative will be computed.
        grad_tensors (Tensor or Sequence[Tensor], optional): The "vector" in the Jacobian-vector
            product, usually gradients each element of corresponding tensors. (None values can be
            specified for scalar Tensors or ones that don't require grad.)
        retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be
            reset after backward is complete. Defaults to ``False``. Note that in nearly all cases
            setting this option to ``True`` is not needed and often can be worked around in a much
            more efficient way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Defaults to ``False``.
    """
    backward_api(
        convert_to_tensor_tuple(tensors),
        convert_to_tensor_tuple(grad_tensors),
        retain_graph,
        create_graph,
    )
