
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

from typing import Union, Sequence, Tuple

from oneflow.python.framework.tensor import Tensor
from oneflow_api import TensorTuple


def convert2tensor_tuple(args: Union[Tensor, Sequence[Tensor], None]):
    if args is None:
        return TensorTuple()
    elif isinstance(args, Tensor):
        if not args.is_determined:
            args.determine()
        tensor_tuple = TensorTuple()
        tensor_tuple.append(args._local_or_consistent_tensor)
        return tensor_tuple
    else:
        for tensor in args:
            if not tensor.is_determined:
                tensor.determine()
        return TensorTuple([x._local_or_consistent_tensor for x in args])
