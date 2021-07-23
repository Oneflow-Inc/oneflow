import collections
from typing import Union, Sequence, Tuple, Optional
from oneflow.framework.tensor import Tensor as PyTensor
from oneflow._oneflow_internal import TensorTuple, Tensor


def convert_to_tensor_tuple(
    args: Optional[Union[PyTensor, Sequence[PyTensor], Tensor, Sequence[Tensor]]]
):
    if args is None:
        return TensorTuple()
    elif isinstance(args, collections.abc.Sequence):
        if isinstance(args[0], PyTensor):
            for tensor in args:
                if not tensor.is_determined:
                    tensor.determine()
            return TensorTuple([x._local_or_consistent_tensor for x in args])
        return TensorTuple(args)
    else:
        tensor_tuple = TensorTuple()
        if isinstance(args, PyTensor):
            if not args.is_determined:
                args.determine()
            tensor_tuple.append(args._local_or_consistent_tensor)
        else:
            tensor_tuple.append(args)
        return tensor_tuple
