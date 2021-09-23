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

oneflow.sbp.sbp.__doc__ = r"""
    A sbp is an object representing the distribution type of a oneflow.Tensor around the device group,
    which represents the mapping relationship between the logical Tensor and the physical Tensor.
    
    sbp includes three types:
    
    1. split: 
        Indicates that the physical Tensors are obtained by splitting the logical Tensor.
        Split will contain a parameter Axis, which represents the dimension to be split.
        If all the physical Tensors are spliced according to the dimensions of Split,
        the logical Tensor can be restored.
    
    2. broadcast: 
        Indicates that the physical Tensors are copies of the logical Tensor, which are
        exactly the same.
    
    3. partial_sum: 
        Indicates that the physical Tensors have the same shape as the logical Tensor,
        but the value of the element at each corresponding position is a part of the
        value of the element at the corresponding position of the logical Tensor. The
        logical Tensor can be returned by adding all the physical Tensors according to
        the corresponding positions (element-wise).
    
    A oneflow.Tensor's sbp can be accessed via the Tensor.sbp property.
    
    A sbp can be constructed in several ways:
    
    .. code-block:: python

        >>> import oneflow as flow
        
        >>> s = flow.sbp.split(0)
        >>> s
        oneflow.sbp.split(axis=0)
        >>> b = flow.sbp.broadcast
        >>> b
        oneflow.sbp.broadcast
        >>> p = flow.sbp.partial_sum
        >>> p
        oneflow.sbp.partial_sum
    
    """
