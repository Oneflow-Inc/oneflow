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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

oneflow.placement.__doc__ = r"""
    A ``oneflow.placement`` is an object representing the device group on which a :class:`oneflow.Tensor` is or will be allocated. The ``oneflow.placement`` contains a device type ('cpu' or 'cuda') and corresponding device sequence.
    
    A :class:`oneflow.Tensor`'s placement can be accessed via the Tensor.placement property.
    
    A oneflow.placement can be constructed in several ways:
    
    .. code-block:: python

        >>> import oneflow as flow
        
        >>> p = flow.placement(type="cuda", ranks=[0, 1, 2, 3])
        >>> p
        oneflow.placement(type="cuda", ranks=[0, 1, 2, 3])
        >>> p = flow.placement(type="cuda", ranks=[[0, 1], [2, 3]])
        >>> p
        oneflow.placement(type="cuda", ranks=[[0, 1], [2, 3]])
        
    """

oneflow.sbp.sbp.__doc__ = r"""
    A ``oneflow.sbp`` is an object representing that how the data of the global tensor is distributed across the ranks of the ``Tensor`` placement.

    ``oneflow.sbp`` includes three types:

        - oneflow.sbp.split(axis)

          Indicates that the global tensor is evenly divided according to the dimension `axis` and distributed on each rank.

        - oneflow.sbp.broadcast()

          Indicates that the global tensor is replicated on each rank.

        - oneflow.sbp.partial_sum()

          Indicates that the value of the global tensor is element-wise sum of the local tensors distributed in each rank.


    A :class:`oneflow.Tensor`'s sbp can be accessed via the Tensor.sbp property.

    A ``oneflow.sbp`` can be constructed in several ways:

    .. code-block:: python

        >>> import oneflow as flow

        >>> s = flow.sbp.split(0)
        >>> s
        oneflow.sbp.split(axis=0)
        >>> b = flow.sbp.broadcast()
        >>> b
        oneflow.sbp.broadcast
        >>> p = flow.sbp.partial_sum()
        >>> p
        oneflow.sbp.partial_sum
    """
