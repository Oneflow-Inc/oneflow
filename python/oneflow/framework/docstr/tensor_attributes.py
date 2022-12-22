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
from oneflow.framework.docstr.utils import add_docstr, reset_docstr

oneflow.device.__doc__ = r"""
    A :class:`oneflow.device` is an object representing the device on which a :class:`oneflow.Tensor` is or will be allocated.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/tensor_attributes.html#torch.torch.device.
    
    The :class:`oneflow.device` contains a device type ('cpu' or 'cuda') and optional device ordinal for the device type. If the 
    device ordinal is not present, this object will always represent the current device for the device type.

    A :class:`oneflow.device`’s device can be accessed via the Tensor.device property.

    A :class:`oneflow.device` can be constructed via a string or via a string and device ordinal

    Via a string:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.device('cuda:0')
        device(type='cuda', index=0)

        >>> flow.device('cpu')
        device(type='cpu', index=0)

        >>> flow.device('cuda')  # current cuda device
        device(type='cuda', index=0)
    
    Via a string and device ordinal:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.device('cuda', 0)
        device(type='cuda', index=0)

        >>> flow.device('cpu', 0)
        device(type='cpu', index=0)
    
    Note:
        The :class:`oneflow.device` argument in functions can generally be substituted with a string. This allows for fast prototyping of code.
        
        .. code-block:: python

            >>> import oneflow as flow
            >>> # Example of a function that takes in a oneflow.device
            >>> cuda0 = flow.device('cuda:0')
            >>> x = flow.randn(2,3, device=cuda0)
        
        .. code-block:: python

            >>> # You can substitute the flow.device with a string
            >>> x = flow.randn(2,3, device='cuda:0')

"""


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

reset_docstr(
    oneflow.placement.all,
    r"""
    oneflow.placement.all(device_type) -> oneflow.placement

    Returns a placement that contains all available devices.

    Args:
        device_type (str): cuda or cpu

    For examples:

    .. code-block:: python

        # Runs on 4 ranks
        import oneflow as flow

        p = flow.placement.all("cuda") # oneflow.placement(type="cuda", ranks=[0, 1, 2, 3])
        p = flow.placement.all("cpu") # oneflow.placement(type="cpu", ranks=[0, 1, 2, 3])

    """,
)

oneflow.sbp.sbp.__doc__ = r"""
    A ``oneflow.sbp`` is an object representing that how the data of the global tensor is distributed across the ranks of the ``Tensor`` placement.

    ``oneflow.sbp`` includes three types:

        - oneflow.sbp.split(dim)

          Indicates that the global tensor is evenly divided according to the dimension `dim` and distributed on each rank.

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
        oneflow.sbp.split(dim=0)
        >>> b = flow.sbp.broadcast()
        >>> b
        oneflow.sbp.broadcast
        >>> p = flow.sbp.partial_sum()
        >>> p
        oneflow.sbp.partial_sum
    """
