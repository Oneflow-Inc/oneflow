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

add_docstr(
    oneflow.tensor,
    r"""
    Constructs a tensor with data, return a global tensor if placement and sbp are in kwargs,
       otherwise return a local tensor.

    Arguments:
        data: Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar or tensor.
    Keyword Arguments:
        dtype (oneflow.dtype, optional) – the desired data type of returned tensor.
            Default: if None, infers data type from data.
        device (oneflow.device, optional): the desired device of returned tensor. If placement
            and sbp is None, uses the current cpu for the default tensor type.
        placement (oneflow.placement, optional): the desired placement of returned tensor.
        sbp (oneflow.sbp or tuple of oneflow.sbp, optional): the desired sbp of returned tensor.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False
        pin_memory(bool, optional): If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: False.

    Note:
        The Keyword Argument device is mutually exclusive with placement and sbp.


    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.from_numpy,
    r"""
    Creates a ``Tensor`` from a ``numpy.ndarray``.

    The returned tensor and ndarray share the same memory. Modifications to the tensor
    will be reflected in the ndarray and vice versa.

    It currently accepts ndarray with dtypes of numpy.float64, numpy.float32, numpy.float16,
    numpy.int64, numpy.int32, numpy.int8, numpy.uint8.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> np_arr = np.arange(6).reshape(2, 3)
        >>> t = flow.from_numpy(np_arr)
        >>> t
        tensor([[0, 1, 2],
                [3, 4, 5]], dtype=oneflow.int64)
        >>> np_arr[0, 0] = -1
        >>> t
        tensor([[-1,  1,  2],
                [ 3,  4,  5]], dtype=oneflow.int64)
    """,
)


add_docstr(
    oneflow.Tensor.device,
    r"""    
    Is the :class:`oneflow.device` where this Tensor is, which is invalid for global tensor.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.Tensor.device.html.
    """,
)

add_docstr(
    oneflow.Tensor.placement,
    r"""
    Is the :class:`oneflow.placement` where this Tensor is, which is invalid for local tensor.
    """,
)

add_docstr(
    oneflow.Tensor.sbp,
    r"""
    Is the ``oneflow.sbp`` representing that how the data of the global tensor is distributed, which is invalid for local tensor.
    """,
)

add_docstr(
    oneflow.Tensor.is_global,
    r"""
    Return whether this Tensor is a global tensor.
    """,
)

add_docstr(
    oneflow.Tensor.is_lazy,
    r"""
    Return whether this Tensor is a lazy tensor.
    """,
)

add_docstr(
    oneflow.Tensor.atan2,
    r"""
    See :func:`oneflow.atan2`
    """,
)

add_docstr(
    oneflow.Tensor.expand,
    """
    Tensor.expand() -> Tensor

    See :func:`oneflow.expand`
    """,
)

add_docstr(
    oneflow.Tensor.expand_as,
    """
    expand_as(other) -> Tensor

    Expand this tensor to the same size as :attr:`other`.
    ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.

    Please see :meth:`~Tensor.expand` for more information about ``expand``.

    Args:
        other (:class:`oneflow.Tensor`): The result tensor has the same size
            as :attr:`other`.
    """,
)

add_docstr(
    oneflow.Tensor.flatten,
    """
    See :func:`oneflow.flatten`
    """,
)

add_docstr(
    oneflow.Tensor.floor,
    """
    See :func:`oneflow.floor`
    """,
)

add_docstr(
    oneflow.Tensor.floor_,
    """
    See :func:`oneflow.floor_`
    """,
)

add_docstr(
    oneflow.Tensor.flip,
    """
    See :func:`oneflow.flip`
    """,
)

add_docstr(
    oneflow.Tensor.in_top_k,
    """
    Tensor.in_top_k(targets, predictions, k) -> Tensor

    See :func:`oneflow.in_top_k`
    """,
)

add_docstr(
    oneflow.Tensor.index_select,
    """
    Tensor.index_select(dim, index) -> Tensor

    See :func:`oneflow.index_select`
    """,
)

add_docstr(
    oneflow.Tensor.numel,
    """
    See :func:`oneflow.numel`
    """,
)

add_docstr(
    oneflow.Tensor.offload,
    """
    Transfer tensor data from GPU memory back to host (CPU) memory. If the tensor is already in host (CPU) memory, the operation does nothing and gives a warning.
    Note that this operation only changes the storage of the tensor, and the tensor id will not change.

    Note:
    
        Both global tensor and local tensor of oneflow are applicable to this operation.

        Use with :func:`oneflow.Tensor.load` and :func:`oneflow.Tensor.is_offloaded`. 
        The behavior of load() is the opposite of offload(), is_offloaded() returns a boolean indicating whether the tensor has been moved to CPU memory.     

        In addition, support for offloading elements of :func:`oneflow.nn.Module.parameters` is provided.        

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> # local tensor
        >>> x = flow.tensor(np.random.randn(1024, 1024, 100), dtype=flow.float32, device=flow.device("cuda"), )
        >>> before_id = id(x)
        >>> x.offload() # Move the Tensor from the GPU to the CPU
        >>> after_id = id(x)
        >>> after_id == before_id
        True
        >>> x.is_offloaded()
        True
        >>> x.load() # Move the Tensor from the cpu to the gpu
        >>> x.is_offloaded()
        False

    .. code-block:: python

        >>> import oneflow as flow

        >>> # global tensor
        >>> # Run on 2 ranks respectively
        >>> placement = flow.placement("cuda", ranks=[0, 1])
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.randn(1024, 1024, 100, dtype=flow.float32, placement=placement, sbp=sbp) # doctest: +SKIP
        >>> before_id = id(x) # doctest: +SKIP
        >>> x.offload() # doctest: +SKIP
        >>> after_id = id(x) # doctest: +SKIP
        >>> print(after_id == before_id) # doctest: +SKIP
        >>> print(x.is_offloaded()) # doctest: +SKIP
        >>> x.load() # doctest: +SKIP
        >>> print(x.is_offloaded()) # doctest: +SKIP
    """,
)

add_docstr(
    oneflow.Tensor.load,
    """
    Load tensor data stored on the host (CPU) back to GPU memory. If the tensor is already in GPU memory, the operation does nothing and gives a warning.

    """,
)

add_docstr(
    oneflow.Tensor.is_offloaded,
    """
    Tensor.is_offloaded() -> bool

    Determine whether the tensor has been moved to CPU memory and the CUDA device memory has been released.

    """,
)

add_docstr(
    oneflow.Tensor.new_empty,
    """
    Tensor.new_empty(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a Tensor of size :attr:`size` filled with uninitialized data. By default, the returned Tensor has the same :attr:`flow.dtype` and :attr:`flow.device` as this tensor.

    Args:
        size (int...): a list, tuple, or flow.Size of integers defining the shape of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.ones(())
        >>> y = x.new_empty((2, 2))
        >>> y.shape
        oneflow.Size([2, 2])
    """,
)

add_docstr(
    oneflow.Tensor.new_ones,
    """
    Tensor.new_ones() -> Tensor

    See :func:`oneflow.new_ones`
    """,
)

add_docstr(
    oneflow.Tensor.new_zeros,
    """
    Tensor.new_zeros(size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a Tensor of size size filled with 0. By default, the returned Tensor has the same oneflow.dtype, oneflow.device or oneflow.placement and oneflow.sbp as this tensor.

    Args:
        size (int...): a list, tuple, or flow.Size of integers defining the shape of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.Tensor(np.ones((1, 2, 3)))
        >>> y = x.new_zeros((2, 2))
        >>> y
        tensor([[0., 0.],
                [0., 0.]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.new_full,
    """
    Tensor.new_full(size, fill_value, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    Returns a Tensor of size size filled with fill_value. By default, the returned Tensor has the same oneflow.dtype, oneflow.device or oneflow.placement and oneflow.sbp as this tensor.

    Args:
        fill_value (scalar): the number to fill the output tensor with.
        size (int...): a list, tuple, or flow.Size of integers defining the shape of the output tensor.
        dtype (flow.dtype, optional):  the desired type of returned tensor. Default: if None, same flow.dtype as this tensor.
        device (flow.device, optional): the desired device of returned tensor. Default: if None, same flow.device as this tensor.
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp descriptor of returned global tensor. Default: if None, the returned tensor is local one using the argument `device`.
        requires_grad (bool, optional): If autograd should record operations on the returned tensor. Default: False.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> tensor = flow.ones((2,), dtype=flow.float64)
        >>> tensor.new_full((3, 4), 3.141592)
        tensor([[3.1416, 3.1416, 3.1416, 3.1416],
                [3.1416, 3.1416, 3.1416, 3.1416],
                [3.1416, 3.1416, 3.1416, 3.1416]], dtype=oneflow.float64)
    """,
)

add_docstr(
    oneflow.Tensor.storage_offset,
    """
    Tensor.storage_offset() -> Tensor

    Returns self tensor’s offset in the underlying storage in terms of number of storage elements (not bytes).

    Example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3, 4, 5])
        >>> x.storage_offset()
        0
    """,
)

add_docstr(
    oneflow.Tensor.local_to_global,
    """
    Tensor.local_to_global(placement=None, sbp=None, *, check_meta=True, copy=False) -> Tensor

    Creates a global tensor from a local tensor.

    Note:
        This tensor must be local tensor.

        Both placement and sbp are required.

        The returned global tensor takes this tensor as its local component in the current rank.

        There is no data communication usually, but when sbp is ``oneflow.sbp.broadcast``, the data on rank 0 will be broadcast to other ranks.

    .. warning::
        When the sbp is ``oneflow.sbp.broadcast``, the data on the non-0 rank will be modified. If you want to keep the input local tensor unchanged,
        please set the arg copy to True.

    Args:
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: None
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp of returned global tensor. Default: None
    Keyword Args:
        check_meta (bool, optional): indicates whether to check meta information when createing global tensor from local
            tensor. Only can be set to False when the shape and dtype of the input local tensor on each rank are the same. If set to False, the
            execution of local_to_global can be accelerated. Default: True
        copy (bool, optional): When copy is set, the returned global tensor takes the replication of this tensor as its local component in the current rank. Default: False

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32) # doctest: +SKIP
        >>> output = input.local_to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)], check_meta=False) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32) 
 
    .. code-block:: python

        >>> # results on rank 1
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.global_to_global,
    """
    Tensor.global_to_global(placement=None, sbp=None, *, grad_sbp=None, check_meta=False, copy=False) -> Tensor

    Performs Tensor placement and/or sbp conversion.

    Note:
        This tensor must be global tensor.

        At least one of placement and sbp is required.

        If placement and sbp are all the same as this tensor's own placement and sbp, then returns this tensor own.
    
    Args:
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: None
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp of returned global tensor. Default: None
    Keyword Args:
        grad_sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): manually specify the sbp of this tensor's grad
            tensor in the backward pass. If None, the grad tensor sbp will be infered automatically. Default: None
        check_meta (bool, optional): indicates whether to check meta information. If set to True, check the consistency
            of the input meta information (placement and sbp) on each rank. Default: False
        copy (bool, optional): When copy is set, a new Tensor is created even when the Tensor already matches the desired conversion. Default: False

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32, placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.broadcast]) # doctest: +SKIP
        >>> output = input.global_to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)]) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)

    .. code-block:: python

        >>> # results on rank 1
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.to_global,
    """
    Tensor.to_global(placement=None, sbp=None, **kwargs) -> Tensor

    Creates a global tensor if this tensor is a local tensor, otherwise performs Tensor placement and/or sbp conversion.

    Note:
        This tensor can be local tensor or global tensor.

        - For local tensor

          Both placement and sbp are required.

          The returned global tensor takes this tensor as its local component in the current rank.

          There is no data communication usually, but when sbp is ``oneflow.sbp.broadcast``, the data on rank 0 will be broadcast to other ranks.

        - For global tensor

          At least one of placement and sbp is required.

          If placement and sbp are all the same as this tensor's own placement and sbp, then returns this tensor own.

    .. warning::
        When the input tensor is a local tensor and sbp is ``oneflow.sbp.broadcast``, the data on the non-0 rank will be modified.
        If you want to keep the input local tensor unchanged, please set the arg copy to True.

    Args:
        placement (flow.placement, optional): the desired placement of returned global tensor. Default: None
        sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): the desired sbp of returned global tensor. Default: None
    Keyword Args:
        grad_sbp (flow.sbp.sbp or tuple of flow.sbp.sbp, optional): manually specify the sbp of this tensor's grad
            tensor in the backward pass. If None, the grad tensor sbp will be infered automatically. It is only used if this tensor is a
            global tensor. Default: None
        check_meta (bool, optional): indicates whether to check meta information. If set to True, check the input meta
            information on each rank. Default: True if this tensor is a local tensor, False if this tensor is a global tensor
        copy (bool, optional): When copy is set, copy occurres in this operation. For local tensor, the returned global tensor takes the
            replication of this tensor as its local component in the current rank. For global tensor, a new Tensor is created even when
            the Tensor already matches the desired conversion. Default: False

    For local tensor:

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32) # doctest: +SKIP
        >>> output = input.to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)], check_meta=False) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32) 
 
    .. code-block:: python

        >>> # results on rank 1
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)

    For global tensor:

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32, placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.broadcast]) # doctest: +SKIP
        >>> output = input.to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)]) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)

    .. code-block:: python

        >>> # results on rank 1
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(dim=0),), dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.to_consistent,
    """
    This interface is no longer available, please use :func:`oneflow.Tensor.to_global` instead.
    """,
)

add_docstr(
    oneflow.Tensor.to_local,
    """
    Tensor.to_local(**kwargs) -> Tensor

    Returns the local component of this global tensor in the current rank.

    Keyword Args:
        copy (bool, optional): When copy is set, a new replicated tensor of the local component of this global tensor in the current rank is returned. Default: False

    Note:
        This tensor should be a global tensor, and it returns a empty tensor if there is no local component in the current rank.

        No copy occurred in this operation if copy is not set.

    For example:

    .. code-block:: python

        >>> # Run on 2 ranks respectively
        >>> import oneflow as flow
        >>> x = flow.tensor([0., 1.], dtype=flow.float32, placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)]) # doctest: +SKIP
        >>> y = x.to_local() # doctest: +SKIP
        >>> print(y.size()) # doctest: +SKIP
        >>> print(y) # doctest: +SKIP

    .. code-block:: python

        >>> # results on rank 0
        oneflow.Size([1])
        tensor([0.], dtype=oneflow.float32)

    .. code-block:: python

        >>> # results on rank 1
        oneflow.Size([1])
        tensor([1.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.transpose,
    """
    See :func:`oneflow.transpose`
    """,
)

add_docstr(
    oneflow.Tensor.logical_not,
    """
    logical_not() -> Tensor
    See :func:`oneflow.logical_not`
    """,
)

add_docstr(
    oneflow.Tensor.lerp,
    """
    See :func:`oneflow.lerp`
    """,
)

add_docstr(
    oneflow.Tensor.lerp_,
    """
    See :func:`oneflow.lerp_`
    """,
)

add_docstr(
    oneflow.Tensor.quantile,
    """
    See :func:`oneflow.quantile`
    """,
)

add_docstr(
    oneflow.Tensor.sqrt,
    """
    See :func:`oneflow.sqrt`
    """,
)

add_docstr(
    oneflow.Tensor.square,
    """
    See :func:`oneflow.square`
    """,
)

add_docstr(
    oneflow.Tensor.std,
    """
    See :func:`oneflow.std`
    """,
)

add_docstr(
    oneflow.Tensor.var,
    """
    See :func:`oneflow.var`
    """,
)

add_docstr(
    oneflow.Tensor.squeeze,
    """
    Tensor.squeeze(dim=None) -> Tensor
    See :func:`oneflow.squeeze`
    """,
)

add_docstr(
    oneflow.Tensor.squeeze_,
    """
    Tensor.squeeze_(dim=None) -> Tensor
    In-place version of :func:`oneflow.Tensor.squeeze`
    """,
)

add_docstr(
    oneflow.Tensor.unfold,
    """
    Returns a view of the original tensor which contains all slices of `size` size from `self`
    tensor in the dimension `dimension`.

    Step between two slices is given by `step`.

    If sizedim is the size of dimension `dimension` for `self`, the size of dimension dimension in the
    returned tensor will be (sizedim - size) / step + 1.

    An additional dimension of size `size` is appended in the returned tensor.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.Tensor.unfold.html.

    Args:
        dimension (int): dimension in which unfolding happens
        size (int): the size of each slice that is unfolded
        step (int): the step between each slice

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.arange(1, 8)
        >>> x
        tensor([1, 2, 3, 4, 5, 6, 7], dtype=oneflow.int64)
        >>> x.unfold(0, 2, 1)
        tensor([[1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7]], dtype=oneflow.int64)
        >>> x.unfold(0, 2, 2)
        tensor([[1, 2],
                [3, 4],
                [5, 6]], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.Tensor.matmul,
    """
    See :func:`oneflow.matmul`
    """,
)

add_docstr(
    oneflow.Tensor.mv,
    """
    See :func:`oneflow.mv`
    """,
)

add_docstr(
    oneflow.Tensor.mm,
    """
    See :func:`oneflow.mm`
    """,
)

add_docstr(
    oneflow.Tensor.narrow,
    """
    See :func:`oneflow.narrow`
    """,
)

add_docstr(
    oneflow.Tensor.unsqueeze,
    """
    Tensor.unsqueeze(dim) -> Tensor

    See :func:`oneflow.unsqueeze`
    """,
)

add_docstr(
    oneflow.Tensor.unsqueeze_,
    """
    Tensor.unsqueeze_(dim) -> Tensor

    In-place version of :func:`oneflow.Tensor.unsqueeze`
    """,
)

add_docstr(
    oneflow.Tensor.as_strided,
    """
    Tensor.as_strided(size, stride, storage_offset=None) -> Tensor

    See :func:`oneflow.as_strided`
    """,
)

add_docstr(
    oneflow.Tensor.as_strided_,
    """
    Tensor.as_strided_(size, stride, storage_offset=None) -> Tensor

    In-place version of :func:`oneflow.Tensor.as_strided`
    """,
)

add_docstr(
    oneflow.Tensor.permute,
    """
    See :func:`oneflow.permute`
    """,
)

add_docstr(
    oneflow.Tensor.abs,
    """
    See :func:`oneflow.abs`
    """,
)

add_docstr(
    oneflow.Tensor.acos,
    """
    See :func:`oneflow.acos`
    """,
)

add_docstr(
    oneflow.Tensor.arccos,
    """
    See :func:`oneflow.arccos`
    """,
)

add_docstr(
    oneflow.Tensor.acosh,
    """
    See :func:`oneflow.acosh`
    """,
)

add_docstr(
    oneflow.Tensor.arccosh,
    """
    See :func:`oneflow.arccosh`
    """,
)

add_docstr(
    oneflow.Tensor.arctanh,
    """
    See :func:`oneflow.arctanh`
    """,
)

add_docstr(
    oneflow.Tensor.argmax,
    """
    See :func:`oneflow.argmax`
    """,
)

add_docstr(
    oneflow.Tensor.argmin,
    """
    See :func:`oneflow.argmin`
    """,
)

add_docstr(
    oneflow.Tensor.argsort,
    """
    See :func:`oneflow.argsort`
    """,
)

add_docstr(
    oneflow.Tensor.argwhere,
    """
    See :func:`oneflow.argwhere`
    """,
)

add_docstr(
    oneflow.Tensor.atanh,
    """
    See :func:`oneflow.atanh`
    """,
)

add_docstr(
    oneflow.Tensor.backward,
    """
    Computes the gradient of current tensor `w.r.t.` graph leaves.

    The graph is differentiated using the chain rule. If the tensor is non-scalar (i.e. its data has more than one element) and requires gradient, the function additionally requires specifying gradient. It should be a tensor of matching type and location, that contains the gradient of the differentiated function w.r.t. self.

    This function accumulates gradients in the leaves - you might need to zero .grad attributes or set them to None before calling it. See Default gradient layouts for details on the memory layout of accumulated gradients.

    Note:
        If you run any forward ops, create gradient, and/or call backward in a user-specified CUDA stream context, see Stream semantics of backward passes.
    Note:
        When inputs are provided and a given input is not a leaf, the current implementation will call its grad_fn (though it is not strictly needed to get this gradients). It is an implementation detail on which the user should not rely. See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.Tensor.backward.html.

    Args:
        gradient (Tensor or None): Gradient w.r.t. the tensor. If it is a tensor, it will be automatically converted to a Tensor that does not require grad unless create_graph is True. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable then this argument is optional.

        retain_graph (bool, optional): If False, the graph used to compute the grads will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

        create_graph (bool, optional): If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to False.
    """,
)

add_docstr(
    oneflow.Tensor.grad,
    r"""
    Return the gradient calculated by autograd functions. This property is None by default.
    """,
)

add_docstr(
    oneflow.Tensor.grad_fn,
    r"""
    Return the function that created this tensor if it's ``requires_grad`` is True.
    """,
)

add_docstr(
    oneflow.Tensor.inverse,
    """
    See :func:`oneflow.linalg.inv`
    """,
)

add_docstr(
    oneflow.Tensor.trunc,
    """
    See :func:`oneflow.trunc`
    """,
)

add_docstr(
    oneflow.Tensor.is_leaf,
    r"""
    All Tensors that have ``requires_grad`` which is ``False`` will be leaf Tensors by convention.

    For Tensor that have ``requires_grad`` which is ``True``, they will be leaf Tensors if they
    were created by source operations.

    Only leaf Tensors will have their ``grad`` populated during a call to ``backward()``. To get
    ``grad`` populated for non-leaf Tensors, you can use ``retain_grad()``.

    Compatible with PyTorch.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.rand(10, requires_grad=False)
        >>> a.is_leaf
        True
        >>> a = flow.rand(10, requires_grad=True)
        >>> a.is_leaf
        True
        >>> b = a.cuda()
        >>> b.is_leaf
        False
        >>> c = a + 2
        >>> c.is_leaf
        False
    """,
)

add_docstr(
    oneflow.Tensor.requires_grad,
    r"""
    Is ``True`` if gradient need to be computed for this Tensor, ``False`` otherwise.

    Compatible with PyTorch.
    """,
)

add_docstr(
    oneflow.Tensor.requires_grad_,
    r"""oneflow.Tensor.requires_grad_(requires_grad=True) -> Tensor
    Sets this tensor’s requires_grad attribute in-place. Returns this tensor.

    Compatible with PyTorch.

    Args:
        requires_grad (bool): Change the requires_grad flag for this Tensor. Default is ``True``.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.rand(10, requires_grad=False)
        >>> a.requires_grad
        False
        >>> a = a.requires_grad_(requires_grad=True)
        >>> a.requires_grad
        True
    """,
)

add_docstr(
    oneflow.Tensor.register_hook,
    r"""oneflow.Tensor.register_hook(hook)

    Registers a backward hook.

    The hook will be called every time a gradient with respect to the Tensor is computed.
    The hook should have the following signature:

    .. code-block:: 

        hook(grad) -> Tensor or None


    The hook should not modify its argument, but it can optionally return a new gradient which
    will be used in place of ``grad``.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(5, requires_grad=True)
        >>> def hook(grad):
        ...     return grad * 2
        >>> x.register_hook(hook)
        >>> y = x * 2
        >>> y.sum().backward()
        >>> x.grad
        tensor([4., 4., 4., 4., 4.], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.retain_grad,
    r"""
    Enables this Tensor to have their ``grad`` populated during ``backward()``. This is a no-op
    for leaf tensors.

    Compatible with PyTorch.
    """,
)

add_docstr(
    oneflow.Tensor.bmm,
    """
    See :func:`oneflow.bmm`
    """,
)

add_docstr(
    oneflow.Tensor.chunk,
    """
    See :func:`oneflow.chunk`
    """,
)

add_docstr(
    oneflow.Tensor.split,
    """
    See :func:`oneflow.split`
    """,
)

add_docstr(
    oneflow.Tensor.unbind,
    """
    See :func:`oneflow.unbind`
    """,
)

add_docstr(
    oneflow.Tensor.swapaxes,
    """
    See :func:`oneflow.swapaxes`
    """,
)

add_docstr(
    oneflow.Tensor.amax,
    """
    See :func:`oneflow.amax`
    """,
)

add_docstr(
    oneflow.Tensor.swapdims,
    """
    See :func:`oneflow.swapdims`
    """,
)

add_docstr(
    oneflow.Tensor.cast,
    """
    See :func:`oneflow.cast`
    """,
)

add_docstr(
    oneflow.Tensor.diag,
    """
    See :func:`oneflow.diag`
    """,
)

add_docstr(
    oneflow.Tensor.addcdiv,
    """
    See :func:`oneflow.addcdiv`
    """,
)

add_docstr(
    oneflow.Tensor.addcdiv_,
    """
    In-place version of :func:`oneflow.Tensor.addcdiv`
    """,
)

add_docstr(
    oneflow.Tensor.dim,
    """
    Tensor.dim() → int

    Returns the number of dimensions of self tensor.
    """,
)

add_docstr(
    oneflow.Tensor.element_size,
    """
    Tensor.element_size() → int

    Returns the size in bytes of an individual element.

    """,
)

add_docstr(
    oneflow.Tensor.exp,
    """
    See :func:`oneflow.exp`
    """,
)

add_docstr(
    oneflow.Tensor.exp2,
    """
    See :func:`oneflow.exp2`
    """,
)

add_docstr(
    oneflow.Tensor.erf,
    """
    Tensor.erf() -> Tensor

    See :func:`oneflow.erf`
    """,
)

add_docstr(
    oneflow.Tensor.erfc,
    """
    Tensor.erfc() -> Tensor

    See :func:`oneflow.erfc`
    """,
)

add_docstr(
    oneflow.Tensor.erfinv,
    """
    See :func:`oneflow.erfinv`
    """,
)

add_docstr(
    oneflow.Tensor.erfinv_,
    """
    Inplace version of :func:`oneflow.erfinv`
    """,
)

add_docstr(
    oneflow.Tensor.eq,
    """
    See :func:`oneflow.eq`
    """,
)

add_docstr(
    oneflow.Tensor.equal,
    """
    See :func:`oneflow.equal`
    """,
)

add_docstr(
    oneflow.Tensor.lt,
    """
    See :func:`oneflow.lt`
    """,
)

add_docstr(
    oneflow.Tensor.le,
    """
    See :func:`oneflow.le`
    """,
)

add_docstr(
    oneflow.Tensor.ne,
    """
    See :func:`oneflow.ne`
    """,
)

add_docstr(
    oneflow.Tensor.neg,
    """
    See :func:`oneflow.neg`
    """,
)

add_docstr(
    oneflow.Tensor.norm,
    """
    See :func:`oneflow.norm`
    """,
)

add_docstr(
    oneflow.Tensor.fill_,
    """
    Tensor.fill_(value) → Tensor

    Fills `self` tensor with the specified value.
    """,
)

add_docstr(
    oneflow.Tensor.ge,
    """
    See :func:`oneflow.ge`
    """,
)

add_docstr(
    oneflow.Tensor.get_device,
    """
    Tensor.get_device() -> Device ordinal (Integer)

    For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides. For CPU tensors, an error is thrown.


    """,
)

add_docstr(
    oneflow.Tensor.gt,
    """
    See :func:`oneflow.gt`
    """,
)

add_docstr(
    oneflow.Tensor.gt_,
    """Tensor.gt_(value) -> Tensor
    In-place version of :func:`oneflow.Tensor.gt`.
    """,
)

add_docstr(
    oneflow.Tensor.log1p,
    """
    See :func:`oneflow.log1p`
    """,
)

add_docstr(
    oneflow.Tensor.mish,
    """
    See :func:`oneflow.mish`
    """,
)

add_docstr(
    oneflow.Tensor.mul,
    """Tensor.mul(value) -> Tensor
    See :func:`oneflow.mul`
    """,
)

add_docstr(
    oneflow.Tensor.mul_,
    """Tensor.mul_(value) -> Tensor

    In-place version of :func:`oneflow.Tensor.mul`.
    """,
)

add_docstr(
    oneflow.Tensor.div_,
    """Tensor.div_(value) -> Tensor
    In-place version of :func:`oneflow.Tensor.div`.
    """,
)

add_docstr(
    oneflow.Tensor.sub_,
    """Tensor.sub_(value) -> Tensor
    In-place version of :func:`oneflow.Tensor.sub`.
    """,
)

add_docstr(
    oneflow.Tensor.negative,
    """
    See :func:`oneflow.negative`
    """,
)

add_docstr(
    oneflow.Tensor.nelement,
    """
    Tensor.nelement() → int

    Alias for numel()
    """,
)

add_docstr(
    oneflow.Tensor.normal_,
    """
    normal_(mean=0, std=1, *, generator=None) -> Tensor

    Fills :attr:`self` tensor with elements samples from the normal distribution parameterized by :attr:`mean` and :attr:`std`.
    """,
)

add_docstr(
    oneflow.Tensor.numpy,
    """
    Tensor.numpy() → numpy.ndarray

    Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray share the same underlying storage. Changes to
     self tensor will be reflected in the ndarray and vice versa.
    """,
)

add_docstr(
    oneflow.Tensor.pow,
    """
    See :func:`oneflow.pow`
    """,
)

add_docstr(
    oneflow.Tensor.relu,
    """
    See :func:`oneflow.relu`
    """,
)

add_docstr(
    oneflow.Tensor.roll,
    """
    See :func:`oneflow.roll`
    """,
)

add_docstr(
    oneflow.Tensor.round,
    """
    See :func:`oneflow.round`
    """,
)

add_docstr(
    oneflow.Tensor.round_,
    """
    See :func:`oneflow.round_`
    """,
)

add_docstr(
    oneflow.Tensor.reciprocal,
    """
    See :func:`oneflow.reciprocal`
    """,
)

add_docstr(
    oneflow.Tensor.add,
    """
    See :func:`oneflow.add`
    """,
)

add_docstr(
    oneflow.Tensor.addmm,
    """
    See :func:`oneflow.addmm`
    """,
)

add_docstr(
    oneflow.Tensor.add_,
    """
    In-place version of :func:`oneflow.Tensor.add`.
    """,
)

add_docstr(
    oneflow.Tensor.addcmul,
    """
    See :func:`oneflow.addcmul`
    """,
)

add_docstr(
    oneflow.Tensor.addcmul_,
    """
    In-place version of :func:`oneflow.Tensor.addcmul`.
    """,
)

add_docstr(
    oneflow.Tensor.asin,
    """
    See :func:`oneflow.asin`
    """,
)

add_docstr(
    oneflow.Tensor.asinh,
    """
    See :func:`oneflow.asinh`
    """,
)

add_docstr(
    oneflow.Tensor.arcsin,
    """
    See :func:`oneflow.arcsin`
    """,
)

add_docstr(
    oneflow.Tensor.arcsinh,
    """
    See :func:`oneflow.arcsinh`
    """,
)

add_docstr(
    oneflow.Tensor.sin,
    """
    sin() -> Tensor

    See :func:`oneflow.sin`
    """,
)

add_docstr(
    oneflow.Tensor.sin_,
    """
    See :func:`oneflow.sin_`
    """,
)

add_docstr(
    oneflow.Tensor.cos,
    """
    See :func:`oneflow.cos`
    """,
)

add_docstr(
    oneflow.Tensor.diagonal,
    """
    See :func:`oneflow.diagonal`
    """,
)

add_docstr(
    oneflow.Tensor.log,
    """
    See :func:`oneflow.log`
    """,
)

add_docstr(
    oneflow.Tensor.log2,
    """
    See :func:`oneflow.log2`
    """,
)

add_docstr(
    oneflow.Tensor.log10,
    """
    See :func:`oneflow.log10`
    """,
)

add_docstr(
    oneflow.Tensor.ndim,
    """
    See :func:`oneflow.Tensor.dim`
    """,
)

add_docstr(
    oneflow.Tensor.rsqrt,
    """
    See :func:`oneflow.rsqrt`
    """,
)

add_docstr(
    oneflow.Tensor.cosh,
    """
    See :func:`oneflow.cosh`
    """,
)

add_docstr(
    oneflow.Tensor.atan,
    """
    See :func:`oneflow.atan`
    """,
)

add_docstr(
    oneflow.Tensor.arctan,
    """
    See :func:`oneflow.arctan`
    """,
)

add_docstr(
    oneflow.Tensor.dot,
    """
    See :func:`oneflow.dot`
    """,
)

add_docstr(
    oneflow.Tensor.selu,
    """
    See :func:`oneflow.selu`
    """,
)

add_docstr(
    oneflow.Tensor.sigmoid,
    """
    See :func:`oneflow.sigmoid`
    """,
)

add_docstr(
    oneflow.Tensor.sign,
    """
    See :func:`oneflow.sign`
    """,
)

add_docstr(
    oneflow.Tensor.silu,
    """
    See :func:`oneflow.silu`
    """,
)

add_docstr(
    oneflow.Tensor.sinh,
    """
    See :func:`oneflow.sinh`
    """,
)

add_docstr(
    oneflow.Tensor.size,
    """
    Returns the size of the self tensor. If dim is not specified, the returned value is a oneflow.Size, a subclass of tuple. If dim is specified, returns an int holding the size of that dimension.

    The interface is consistent with PyTorch.

    Args:
        idx (int, optional): The dimension for which to retrieve the size.


    """,
)

add_docstr(
    oneflow.Tensor.softmax,
    """
    See :func:`oneflow.softmax`
    """,
)

add_docstr(
    oneflow.Tensor.softplus,
    """
    See :func:`oneflow.softplus`
    """,
)

add_docstr(
    oneflow.Tensor.softsign,
    """
    See :func:`oneflow.softsign`
    """,
)

add_docstr(
    oneflow.Tensor.tan,
    """
    See :func:`oneflow.tan`
    """,
)

add_docstr(
    oneflow.Tensor.tanh,
    """
    See :func:`oneflow.tanh`
    """,
)

add_docstr(
    oneflow.Tensor.tril,
    """
    See :func:`oneflow.tril`
    """,
)

add_docstr(
    oneflow.Tensor.triu,
    """
    See :func:`oneflow.triu`
    """,
)

add_docstr(
    oneflow.Tensor.uniform_,
    """
    Tensor.uniform_(from=0, to=1) → Tensor

    Fills self tensor with numbers sampled from the continuous uniform distribution:

    .. math::
        P(x)=1/(to-from)

    """,
)

add_docstr(
    oneflow.Tensor.copy_,
    """
    Copies the elements from src into self tensor and returns self.

    The src tensor must be broadcastable with the self tensor. It may be of a different data type or reside on a different device.

    The interface is consistent with PyTorch.

    Args:

        src (Tensor): the source tensor to copy from

        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously with respect to the host. For other cases, this argument has no effect.
    """,
)

add_docstr(
    oneflow.Tensor.to,
    """Performs Tensor dtype and/or device conversion.
        A flow.dtype and flow.device are inferred from the arguments of `input.to(*args, **kwargs)`.

    .. note::
        If the ``input`` Tensor already
        has the correct :class:`flow.dtype` and :class:`flow.device`, then ``input`` is returned.
        Otherwise, the returned tensor is a copy of ``input`` with the desired.

    Args:
        input (oneflow.Tensor): An input tensor.
        *args (oneflow.Tensor or oneflow.device or oneflow.dtype): Positional arguments
        **kwargs (oneflow.device or oneflow.dtype) : Key-value arguments

    Returns:
        oneflow.Tensor: A Tensor.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """,
)


add_docstr(
    oneflow.Tensor.half,
    """
    self.half() is equivalent to self.to(dtype=oneflow.float16).

    See :func:`oneflow.Tensor.to`

    """,
)


add_docstr(
    oneflow.Tensor.gather,
    """
    oneflow.Tensor.gather(dim, index) -> Tensor

    See :func:`oneflow.gather`

    """,
)

add_docstr(
    oneflow.Tensor.clamp,
    """
    See :func:`oneflow.clamp`.
    """,
)

add_docstr(
    oneflow.Tensor.clamp_,
    """
    Inplace version of :func:`oneflow.Tensor.clamp`.
    """,
)

add_docstr(
    oneflow.Tensor.clip,
    """
    Alias for :func:`oneflow.Tensor.clamp`.
    """,
)

add_docstr(
    oneflow.Tensor.clip_,
    """
    Alias for :func:`oneflow.Tensor.clamp_`.
    """,
)

add_docstr(
    oneflow.Tensor.cpu,
    r"""Returns a copy of this object in CPU memory.
    If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 2, 3, 4, 5], device=flow.device("cuda"))
        >>> output = input.cpu()
        >>> output.device
        device(type='cpu', index=0)
    """,
)

add_docstr(
    oneflow.Tensor.cuda,
    r"""Returns a copy of this object in CUDA memory.
    If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.

    Args:
        device  (flow.device): The destination GPU device. Defaults to the current CUDA device.

    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.Tensor([1, 2, 3, 4, 5])
        >>> output = input.cuda()
        >>> output.device
        device(type='cuda', index=0)
    """,
)

add_docstr(
    oneflow.Tensor.cumprod,
    """
    See :func:`oneflow.cumprod`
    """,
)

add_docstr(
    oneflow.Tensor.cumsum,
    """
    See :func:`oneflow.cumsum`
    """,
)

add_docstr(
    oneflow.Tensor.repeat,
    """
    Tensor.repeat(*size) -> Tensor

    See :func:`oneflow.repeat`
    """,
)

add_docstr(
    oneflow.Tensor.repeat_interleave,
    """
    Tensor.repeat_interleave(repeats, dim=None, *, output_size=None) -> Tensor

    See :func:`oneflow.repeat_interleave`
    """,
)

add_docstr(
    oneflow.Tensor.t,
    """
    See :func:`oneflow.t`

    Tensor.t() → Tensor
    """,
)

add_docstr(
    oneflow.Tensor.tile,
    """
    Tensor.tile(*dims) -> Tensor

    See :func:`oneflow.tile`
    """,
)

add_docstr(
    oneflow.Tensor.T,
    """
    Is this Tensor with its dimensions reversed.

    If `n` is the number of dimensions in `x`, `x.T` is equivalent to `x.permute(n-1, n-2, ..., 0)`.
    """,
)

add_docstr(
    oneflow.Tensor.fmod,
    """
    Tensor.fmod(other) -> Tensor

    See :func:`oneflow.fmod`

    """,
)

add_docstr(
    oneflow.Tensor.logical_and,
    """
    logical_and() -> Tensor

    See :func:`oneflow.logical_and`

    """,
)

add_docstr(
    oneflow.Tensor.logical_or,
    """

    logical_or() -> Tensor

    See :func:`oneflow.logical_or`

    """,
)

add_docstr(
    oneflow.Tensor.logical_xor,
    """
    logical_xor() -> Tensor

    See :func:`oneflow.logical_xor`

    """,
)

add_docstr(
    oneflow.Tensor.logsumexp,
    """
    See :func:`oneflow.logsumexp`
    """,
)

add_docstr(
    oneflow.Tensor.masked_fill,
    """
    See :func:`oneflow.masked_fill`
    """,
)

add_docstr(
    oneflow.Tensor.masked_fill_,
    """
    In-place version of :meth:`oneflow.Tensor.masked_fill`.
    """,
)

add_docstr(
    oneflow.Tensor.masked_select,
    """
    See :func:`oneflow.masked_select`
    """,
)

add_docstr(
    oneflow.Tensor.sub,
    """
    See :func:`oneflow.sub`
    """,
)

add_docstr(
    oneflow.Tensor.div,
    """
    See :func:`oneflow.div`

    """,
)

add_docstr(
    oneflow.Tensor.ceil,
    """
    See :func:`oneflow.ceil`
    """,
)

add_docstr(
    oneflow.Tensor.ceil_,
    """
    See :func:`oneflow.ceil_`
    """,
)

add_docstr(
    oneflow.Tensor.expm1,
    """
    See :func:`oneflow.expm1`
    """,
)

add_docstr(
    oneflow.Tensor.topk,
    """
    See :func:`oneflow.topk`
    """,
)

add_docstr(
    oneflow.Tensor.nms,
    """
    See :func:`oneflow.nms`
    """,
)

add_docstr(
    oneflow.Tensor.nonzero,
    """
    nonzero(input, as_tuple=False) -> Tensor

    See :func:`oneflow.nonzero`
    """,
)

add_docstr(
    oneflow.Tensor.max,
    """
    input.max(dim, index) -> Tensor

    See :func:`oneflow.max`
    """,
)

add_docstr(
    oneflow.Tensor.min,
    """
    input.min(dim, index) -> Tensor

    See :func:`oneflow.min`
    """,
)

add_docstr(
    oneflow.Tensor.maximum,
    """
    See :func:`oneflow.maximum`
    """,
)

add_docstr(
    oneflow.Tensor.median,
    """
    See :func:`oneflow.median`
    """,
)

add_docstr(
    oneflow.Tensor.minimum,
    """
    See :func:`oneflow.minimum`
    """,
)

add_docstr(
    oneflow.Tensor.mode,
    """
    See :func:`oneflow.mode`
    """,
)

add_docstr(
    oneflow.Tensor.sum,
    """
    input.sum(dim=None, keepdim=False) -> Tensor

    See :func:`oneflow.sum`
    """,
)

add_docstr(
    oneflow.Tensor.all,
    """
    input.all(dim=None, keepdim=False) -> Tensor

    See :func:`oneflow.all`
    """,
)

add_docstr(
    oneflow.Tensor.any,
    """
    input.any(dim=None, keepdim=False) -> Tensor

    See :func:`oneflow.any`
    """,
)

add_docstr(
    oneflow.Tensor.mean,
    """
    input.mean(dim=None, keepdim=False) -> Tensor

    See :func:`oneflow.mean`
    """,
)

add_docstr(
    oneflow.Tensor.prod,
    """
    input.prod(dim=None, keepdim=False) -> Tensor

    See :func:`oneflow.prod`
    """,
)

add_docstr(
    oneflow.Tensor.reshape,
    """
    See :func:`oneflow.reshape`
    """,
)

add_docstr(
    oneflow.Tensor.reshape_as,
    """
    Tensor.reshape_as(other) -> Tensor
    Returns this tensor as the same shape as other.
    self.reshape_as(other) is equivalent to self.reshape(other.sizes()).
    This method returns a view if other.sizes() is compatible with the current shape.
    See :func:`oneflow.Tensor.view` on when it is possible to return a view.

    Please see reshape() for more information about reshape. See :func:`oneflow.reshape`

    Parameters
    other (oneflow.Tensor) – The result tensor has the same shape as other.
    """,
)

add_docstr(
    oneflow.Tensor.view,
    """
    Returns a new tensor with the same data as the :attr:`self` tensor but of a
    different :attr:`shape`.

    The returned tensor shares the same data and must have the same number
    of elements, but may have a different size. For a tensor to be viewed, the new
    view size must be compatible with its original size and stride, i.e., each new
    view dimension must either be a subspace of an original dimension, or only span
    across original dimensions :math:`d, d+1, \\dots, d+k` that satisfy the following
    contiguity-like condition that :math:`\\forall i = d, \\dots, d+k-1`,

    .. math::

      \\text{stride}[i] = \\text{stride}[i+1] \\times \\text{size}[i+1]

    Otherwise, it will not be possible to view :attr:`self` tensor as :attr:`shape`
    without copying it (e.g., via :meth:`contiguous`). When it is unclear whether a
    :meth:`view` can be performed, it is advisable to use :meth:`reshape`, which
    returns a view if the shapes are compatible, and copies (equivalent to calling
    :meth:`contiguous`) otherwise.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.Tensor.view.html.

    Args:
        input: A Tensor.
        *shape: flow.Size or int...
    Returns:
        A Tensor has the same type as `input`.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = np.array(
        ...    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ... ).astype(np.float32)
        >>> input = flow.Tensor(x)

        >>> y = input.view(2, 2, 2, -1).numpy().shape
        >>> y
        (2, 2, 2, 2)
    """,
)

add_docstr(
    oneflow.Tensor.view_as,
    """
    Tensor.view_as(other) -> Tensor

    Expand this tensor to the same size as :attr:`other`.
    ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.
    
    Please see :meth:`~Tensor.view` for more information about ``view``.

    Args:
        other (:class:`oneflow.Tensor`): The result tensor has the same size
            as :attr:`other`.
    """,
)

add_docstr(
    oneflow.Tensor.sort,
    """
    See :func:`oneflow.sort`
    """,
)

add_docstr(
    oneflow.Tensor.type_as,
    r"""Returns this tensor cast to the type of the given tensor.
        This is a no-op if the tensor is already of the correct type.

    Args:
        input  (Tensor): the input tensor.
        target (Tensor): the tensor which has the desired type.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> target = flow.tensor(np.random.randn(4, 5, 6), dtype = flow.int32)
        >>> input = input.type_as(target)
        >>> input.dtype
        oneflow.int32
    """,
)

add_docstr(
    oneflow.Tensor.bool,
    r"""``Tensor.bool()`` is equivalent to ``Tensor.to(oneflow.bool)``. See :class:`oneflow.Tensor.to()`.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.bool()
        >>> input.dtype
        oneflow.bool

    """,
)

add_docstr(
    oneflow.Tensor.int,
    r"""``Tensor.int()`` is equivalent to ``Tensor.to(flow.int32)``. See :class:`oneflow.Tensor.to()`.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.int()
        >>> input.dtype
        oneflow.int32
    """,
)

add_docstr(
    oneflow.Tensor.long,
    r"""``Tensor.long()`` is equivalent to ``Tensor.to(flow.int64)``. See :class:`oneflow.Tensor.to()`.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64
    """,
)

add_docstr(
    oneflow.Tensor.float,
    r"""``Tensor.float()`` is equivalent to ``Tensor.to(flow.float32)``. See :class:`oneflow.Tensor.to()`.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.int)
        >>> input = input.float()
        >>> input.dtype
        oneflow.float32
    """,
)

add_docstr(
    oneflow.Tensor.double,
    r"""``Tensor.double()`` is equivalent to ``Tensor.to(flow.float64)``. See :class:`oneflow.Tensor.to()`.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.int)
        >>> input = input.double()
        >>> input.dtype
        oneflow.float64
    """,
)

add_docstr(
    oneflow.Tensor.is_contiguous,
    r"""
    Tensor.is_contiguous() -> bool

    Returns True if `self` tensor is contiguous in memory.
    """,
)

add_docstr(
    oneflow.Tensor.is_cuda,
    r"""
    Tensor.is_cuda() -> bool
    
    Is `True` if the Tensor is stored on the GPU, `False` otherwise.
    """,
)

add_docstr(
    oneflow.Tensor.is_floating_point,
    """
    See :func:`oneflow.is_floating_point`
    """,
)

add_docstr(
    oneflow.Tensor.item,
    r"""Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
    For other cases, see tolist().

    This operation is not differentiable.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1.0])
        >>> x.item()
        1.0
    """,
)

add_docstr(
    oneflow.Tensor.tolist,
    r"""Returns the tensor as a (nested) list. For scalars, a standard Python number is returned,
    just like with `item()`. Tensors are automatically moved to the CPU first if necessary.

    This operation is not differentiable.

    Args:
        input  (Tensor): the input tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3], [4,5,6]])
        >>> input.tolist()
        [[1, 2, 3], [4, 5, 6]]
    """,
)

add_docstr(
    oneflow.Tensor.where,
    """
    See :func:`oneflow.where`
    """,
)

add_docstr(
    oneflow.Tensor.zero_,
    r"""
    Tensor.zero_() -> Tensor
    
    Fills `self` tensor with zeros.
    """,
)

add_docstr(
    oneflow.Tensor.isnan,
    """
    See :func:`oneflow.isnan`
    """,
)

add_docstr(
    oneflow.Tensor.isinf,
    """
    See :func:`oneflow.isinf`
    """,
)

add_docstr(
    oneflow.Tensor.byte,
    """
    self.byte() is equivalent to self.to(oneflow.uint8).
    See :func:`oneflow.Tensor.to`
    """,
)

add_docstr(
    oneflow.Tensor.amin,
    """
    See :func:`oneflow.amin`
    """,
)

add_docstr(
    oneflow.Tensor.pin_memory,
    r"""
    Tensor.pin_memory() -> Tensor

    Copies the tensor to pinned memory, if it’s not already pinned.
    """,
)

add_docstr(
    oneflow.Tensor.is_pinned,
    r"""
    Tensor.is_pinned() -> bool

    Returns true if this tensor resides in pinned memory.
    """,
)

add_docstr(
    oneflow.Tensor.type,
    r"""
    type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor

    Returns the type if dtype is not provided, else casts this object to the specified type.

    If this is already of the correct type, no copy is performed and the original object is returned.

    Args:
        dtype (oneflow.dtype or oneflow.tensortype or string, optional): The desired type.
        non_blocking (bool): (**Not Implemented yet**) If True, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed asynchronously with respect to the host.
            Otherwise, the argument has no effect.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.tensor([1, 2], dtype=flow.float32)
        >>> a.type()
        'oneflow.FloatTensor'
        >>> a.type(flow.int8)  # dtype input
        tensor([1, 2], dtype=oneflow.int8)
        >>> a.type(flow.cuda.DoubleTensor)  # tensortype input
        tensor([1., 2.], device='cuda:0', dtype=oneflow.float64)
        >>> a.type("oneflow.HalfTensor")  # string input
        tensor([1., 2.], dtype=oneflow.float16)
    """,
)

add_docstr(
    oneflow.Tensor.scatter,
    """
    See :func:`oneflow.scatter`
    """,
)

add_docstr(
    oneflow.Tensor.scatter_,
    """
    Inplace version of :func:`oneflow.Tensor.scatter`
    """,
)

add_docstr(
    oneflow.Tensor.scatter_add,
    """
    See :func:`oneflow.scatter_add`
    """,
)

add_docstr(
    oneflow.Tensor.scatter_add_,
    """
    Inplace version of :func:`oneflow.Tensor.scatter_add`
    """,
)

add_docstr(
    oneflow.Tensor.cross,
    """
    See :func:`oneflow.cross`
    """,
)

add_docstr(
    oneflow.Tensor.nansum,
    """
    See :func:`oneflow.nansum`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1., 2., float("nan")])
        >>> x.nansum()
        tensor(3., dtype=oneflow.float32)
        >>> x = flow.tensor([[1., float("nan")], [float("nan"), 2]])
        >>> x.nansum(dim=1, keepdim=True)
        tensor([[1.],
                [2.]], dtype=oneflow.float32)
    """,
)

add_docstr(
    oneflow.Tensor.bincount,
    """
    See :func:`oneflow.bincount`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.Tensor([0, 2, 3]).int()
        >>> x.bincount()
        tensor([1, 0, 1, 1], dtype=oneflow.int64)
        >>> weight = flow.Tensor([0.1, 0.2, 0.3])
        >>> x.bincount(weight)
        tensor([0.1000, 0.0000, 0.2000, 0.3000], dtype=oneflow.float32)
        >>> x.bincount(weight, minlength=5)
        tensor([0.1000, 0.0000, 0.2000, 0.3000, 0.0000], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.Tensor.bernoulli,
    """
    See :func:`oneflow.bernoulli`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.Tensor([1, 1, 1])
        >>> x.bernoulli()
        tensor([1., 1., 1.], dtype=oneflow.float32)
        >>> x.bernoulli(p=0.0)
        tensor([0., 0., 0.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.Tensor.bernoulli_,
    """
    The inplace version of :func:`oneflow.Tensor.bernoulli_`.

    See :func:`oneflow.Tensor.bernoulli`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.Tensor([1, 1, 1])
        >>> x.bernoulli_(p=0.0)
        tensor([0., 0., 0.], dtype=oneflow.float32)
        >>> x
        tensor([0., 0., 0.], dtype=oneflow.float32)

    """,
)

add_docstr(
    oneflow.Tensor.broadcast_to,
    """
    See :func:`oneflow.broadcast_to`
    """,
)

add_docstr(
    oneflow.Tensor.unique,
    """
    See :func:`oneflow.unique`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([3, 1, 2, 0 ,2])
        >>> x.unique()
        tensor([0, 1, 2, 3], dtype=oneflow.int64)
        >>> x, indices = x.unique(return_inverse=True)
        >>> indices
        tensor([3, 1, 2, 0, 2], dtype=oneflow.int32)
        >>> x, counts = x.unique(return_counts=True)
        >>> counts
        tensor([1, 1, 1, 1], dtype=oneflow.int32)
    """,
)

add_docstr(
    oneflow.Tensor.clone,
    """
    See :func:`oneflow.clone`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> x.clone()
        tensor([1, 2, 3], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.Tensor.bitwise_and,
    """
    See :func:`oneflow.bitwise_and`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> x.bitwise_and(4)
        tensor([0, 0, 0], dtype=oneflow.int64)
        >>> y = flow.tensor([2, 1, 0])
        >>> x.bitwise_and(y)
        tensor([0, 0, 0], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.Tensor.bitwise_or,
    """
    See :func:`oneflow.bitwise_or`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> x.bitwise_or(4)
        tensor([5, 6, 7], dtype=oneflow.int64)
        >>> y = flow.tensor([2, 1, 0])
        >>> x.bitwise_or(y)
        tensor([3, 3, 3], dtype=oneflow.int64)

    """,
)

add_docstr(
    oneflow.Tensor.bitwise_xor,
    """
    See :func:`oneflow.bitwise_xor`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1, 2, 3])
        >>> x.bitwise_xor(4)
        tensor([5, 6, 7], dtype=oneflow.int64)
        >>> y = flow.tensor([2, 1, 0])
        >>> x.bitwise_xor(y)
        tensor([3, 3, 3], dtype=oneflow.int64)
    """,
)

add_docstr(
    oneflow.Tensor.new,
    """
    Constructs a new tensor of the same data type and device (or placemant and sbp) as self tensor.

    Any valid argument combination to the tensor constructor is accepted by this method,
    including sizes, NumPy ndarray, Python Sequence, etc. See :func:`oneflow.Tensor` for more details.


    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3, 2)
        >>> x.new()
        tensor([], dtype=oneflow.float32)
        >>> x.new(1, 2).shape
        oneflow.Size([1, 2])
        >>> x.new([1, 2])
        tensor([1., 2.], dtype=oneflow.float32)
        >>> y = flow.randn(3, 3)
        >>> x.new(y).shape
        oneflow.Size([3, 3])

    .. warning::
        When y is global tensor, the invoking ``Tensor.new(y)`` will raise an error.
        Consider use ``Tensor.new(y.size())`` to create a tensor that has
        the same placement and sbp with Tensor and the same size with ``y``.

    """,
)

add_docstr(
    oneflow.Tensor.baddbmm,
    """
    See :func:`oneflow.baddbmm`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(2, 3, 4)
        >>> batch1 = flow.randn(2, 3, 5)
        >>> batch2 = flow.randn(2, 5, 4)
        >>> x.baddbmm(batch1, batch2, alpha=2, beta=2) # doctest: +SKIP
    """,
)


add_docstr(
    oneflow.Tensor.frac,
    r"""
    See :func:`oneflow.frac`.
    """,
)

add_docstr(
    oneflow.Tensor.frac_,
    r"""
    In-place version of :func:`oneflow.Tensor.frac`.
    """,
)
