oneflow.Tensor
===================================

.. The documentation is referenced from: 
   https://pytorch.org/docs/1.10/tensors.html

A :class:`oneflow.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

.. currentmodule:: oneflow

Data types
----------

OneFlow defines 8 Tensor types with CPU and GPU variants which are as follows:

======================================= =============================================== =============================== ==================================
Data type                               dtype                                           CPU tensor                      GPU tensor
======================================= =============================================== =============================== ==================================
Boolean                                 ``oneflow.bool``                                :class:`oneflow.BoolTensor`     :class:`oneflow.cuda.BoolTensor`
8-bit integer (unsigned)                ``oneflow.uint8``                               :class:`oneflow.ByteTensor`     :class:`oneflow.cuda.ByteTensor`
8-bit integer (signed)                  ``oneflow.int8``                                :class:`oneflow.CharTensor`     :class:`oneflow.cuda.CharTensor`
64-bit floating point                   ``oneflow.float64`` or ``oneflow.double``       :class:`oneflow.DoubleTensor`   :class:`oneflow.cuda.DoubleTensor`
32-bit floating point                   ``oneflow.float32`` or ``oneflow.float``        :class:`oneflow.FloatTensor`    :class:`oneflow.cuda.FloatTensor`
16-bit floating point                   ``oneflow.float16`` or ``oneflow.half``         :class:`oneflow.HalfTensor`     :class:`oneflow.cuda.HalfTensor`
32-bit integer (signed)                 ``oneflow.int32`` or ``oneflow.int``            :class:`oneflow.IntTensor`      :class:`oneflow.cuda.IntTensor`
64-bit integer (signed)                 ``oneflow.int64`` or ``oneflow.long``           :class:`oneflow.LongTensor`     :class:`oneflow.cuda.LongTensor`
======================================= =============================================== =============================== ==================================

Initializing and basic operations
---------------------------------

A tensor can be constructed from a Python :class:`list` or sequence using the
:func:`oneflow.tensor` constructor:

::

    >>> import oneflow
    >>> import numpy as np
    >>> oneflow.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1., -1.],
            [ 1., -1.]], dtype=oneflow.float32)
    >>> oneflow.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1, 2, 3],
            [ 4, 5, 6]], dtype=oneflow.int64)

.. warning::

    :func:`oneflow.tensor` always copies :attr:`data`. If you have a Tensor
    :attr:`data` and just want to change its ``requires_grad`` flag, use
    :meth:`~oneflow.Tensor.requires_grad_` or
    :meth:`~oneflow.Tensor.detach` to avoid a copy.
    If you have a numpy array and want to avoid a copy, use
    :func:`oneflow.as_tensor`.

.. A tensor of specific data type can be constructed by passing a :class:`oneflow.dtype` and/or a :class:`oneflow.device` to a constructor or tensor creation op:

::

    >>> import oneflow
    >>> oneflow.zeros([2, 4], dtype=oneflow.int32)
    tensor([[ 0, 0, 0, 0],
            [ 0, 0, 0, 0]], dtype=oneflow.int32)
    >>> cuda0 = oneflow.device('cuda:0')
    >>> oneflow.ones([2, 4], dtype=oneflow.float64, device=cuda0)
    tensor([[ 1., 1., 1., 1.],
            [ 1., 1., 1., 1.]], device='cuda:0', dtype=oneflow.float64)

For more information about building tensors, see :ref:`tensor-creation-ops`

The contents of a tensor can be accessed and modified using Python's indexing
and slicing notation:

::

    >>> import oneflow
    >>> x = oneflow.tensor([[1, 2, 3], [4, 5, 6]])
    >>> print(x[1][2])
    tensor(6, dtype=oneflow.int64)
    >>> x[0][1] = 8
    >>> print(x)
    tensor([[1, 8, 3],
            [4, 5, 6]], dtype=oneflow.int64)

Use :meth:`oneflow.Tensor.item` to get a Python number from a tensor containing a
single value:

::

    >>> import oneflow
    >>> x = oneflow.tensor([[1]])
    >>> x
    tensor([[1]], dtype=oneflow.int64)
    >>> x.item()
    1
    >>> x = oneflow.tensor(2.5)
    >>> x
    tensor(2.5000, dtype=oneflow.float32)
    >>> x.item()
    2.5

For more information about indexing, see :ref:`indexing-slicing-joining`

A tensor can be created with :attr:`requires_grad=True` so that
:mod:`oneflow.autograd` records operations on them for automatic differentiation.

::

    >>> import oneflow
    >>> x = oneflow.tensor([[1., -1.], [1., 1.]], requires_grad=True)
    >>> out = x.pow(2).sum()
    >>> out.backward()
    >>> x.grad
    tensor([[ 2., -2.],
            [ 2.,  2.]], dtype=oneflow.float32)

.. note::
   For more information on the :class:`oneflow.dtype`, :class:`oneflow.device`, and
   :class:`oneflow.layout` attributes of a :class:`oneflow.Tensor`, see
   :ref:`tensor-attributes-doc`.

.. note::
   Methods which mutate a tensor are marked with an underscore suffix.
   For example, :func:`oneflow.FloatTensor.add_` computes the absolute value
   in-place and returns the modified tensor, while :func:`oneflow.FloatTensor.add`
   computes the result in a new tensor.

.. note::
    To change an existing tensor's :class:`oneflow.device` and/or :class:`oneflow.dtype`, consider using
    :meth:`~oneflow.Tensor.to` method of Tensor object.

.. warning::
   Current implementation of :class:`oneflow.Tensor` introduces memory overhead,
   thus it might lead to unexpectedly high memory usage in the applications with many tiny tensors.
   If this is your case, consider using one large structure.

Tensor class reference
----------------------

.. class:: Tensor()

   There are a few main ways to create a tensor, depending on your use case.

   - To create a tensor with pre-existing data, use :func:`oneflow.tensor`.
   - To create a tensor with specific size, use ``oneflow.*`` tensor creation
     ops (see :ref:`tensor-creation-ops`).
   - To create a tensor with the same size (and similar types) as another tensor,
     use ``oneflow.*_like`` tensor creation ops
     (see :ref:`tensor-creation-ops`).

.. currentmodule:: oneflow
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    Tensor.new_empty
    Tensor.new_ones 
    Tensor.new_zeros
    Tensor.new_full
    Tensor.new_tensor
    Tensor.is_cuda
    Tensor.is_global
    Tensor.device
    Tensor.grad
    Tensor.ndim
    Tensor.abs
    Tensor.acos
    Tensor.acosh
    Tensor.add
    Tensor.add_
    Tensor.addcdiv
    Tensor.addcdiv_
    Tensor.addcmul
    Tensor.addcmul_
    Tensor.addmm
    Tensor.all
    Tensor.amin
    Tensor.amax
    Tensor.any
    Tensor.arccos
    Tensor.arccosh
    Tensor.arcsin
    Tensor.arcsinh
    Tensor.arctan
    Tensor.arctanh
    Tensor.argmax
    Tensor.argmin
    Tensor.argsort
    Tensor.argwhere
    Tensor.asin
    Tensor.asinh
    Tensor.atan
    Tensor.atan2
    Tensor.atanh
    Tensor.backward
    Tensor.bmm
    Tensor.bool
    Tensor.byte
    Tensor.cast
    Tensor.ceil
    Tensor.ceil_
    Tensor.chunk
    Tensor.clamp
    Tensor.clamp_
    Tensor.clip
    Tensor.clip_
    Tensor.clone
    Tensor.contiguous
    Tensor.copy_
    Tensor.cos
    Tensor.cosh
    Tensor.cpu
    Tensor.cuda
    Tensor.cumprod
    Tensor.cumsum
    Tensor.data
    Tensor.dot
    Tensor.detach
    Tensor.placement
    Tensor.sbp
    Tensor.diag
    Tensor.diagonal
    Tensor.dim
    Tensor.div
    Tensor.div_
    Tensor.double
    Tensor.dtype 
    Tensor.element_size
    Tensor.eq
    Tensor.equal
    Tensor.erf
    Tensor.erfc
    Tensor.erfinv
    Tensor.erfinv_
    Tensor.exp
    Tensor.exp2
    Tensor.expand
    Tensor.expand_as
    Tensor.expm1
    Tensor.fill_
    Tensor.flatten
    Tensor.flip
    Tensor.float
    Tensor.floor
    Tensor.floor_
    Tensor.floor_divide
    Tensor.fmod
    Tensor.gather
    Tensor.ge
    Tensor.get_device
    Tensor.grad_fn
    Tensor.gt
    Tensor.gt_
    Tensor.half
    Tensor.in_top_k
    Tensor.index_select
    Tensor.index_add
    Tensor.index_add_
    Tensor.int
    Tensor.is_contiguous
    Tensor.is_floating_point
    Tensor.is_lazy
    Tensor.is_leaf
    Tensor.isinf
    Tensor.isnan
    Tensor.item
    Tensor.le
    Tensor.lerp
    Tensor.lerp_
    Tensor.log
    Tensor.log1p
    Tensor.log2
    Tensor.log10
    Tensor.logical_and
    Tensor.logical_or
    Tensor.logical_not
    Tensor.logical_xor
    Tensor.long
    Tensor.lt
    Tensor.masked_fill
    Tensor.masked_fill_
    Tensor.masked_select
    Tensor.matmul
    Tensor.mm
    Tensor.mv
    Tensor.max
    Tensor.maximum
    Tensor.median
    Tensor.mean
    Tensor.min
    Tensor.minimum
    Tensor.mish
    Tensor.mode
    Tensor.mul
    Tensor.mul_
    Tensor.frac
    Tensor.frac_
    Tensor.nansum
    Tensor.narrow
    Tensor.ndimension
    Tensor.ne
    Tensor.neg
    Tensor.negative
    Tensor.nelement
    Tensor.nonzero
    Tensor.norm
    Tensor.normal_
    Tensor.numel
    Tensor.numpy
    Tensor.offload
    Tensor.load
    Tensor.is_offloaded
    Tensor.permute
    Tensor.pow
    Tensor.prod
    Tensor.quantile
    Tensor.reciprocal
    Tensor.register_hook
    Tensor.relu
    Tensor.repeat
    Tensor.repeat_interleave
    Tensor.requires_grad
    Tensor.requires_grad_
    Tensor.reshape
    Tensor.reshape_as
    Tensor.retain_grad
    Tensor.roll
    Tensor.round
    Tensor.round_
    Tensor.rsqrt
    Tensor.selu
    Tensor.shape
    Tensor.sigmoid
    Tensor.sign
    Tensor.silu
    Tensor.sin
    Tensor.sin_
    Tensor.sinh
    Tensor.size
    Tensor.softmax
    Tensor.softplus
    Tensor.softsign
    Tensor.sort
    Tensor.split
    Tensor.sqrt
    Tensor.square
    Tensor.squeeze
    Tensor.squeeze_
    Tensor.std
    Tensor.storage_offset
    Tensor.stride
    Tensor.logsumexp
    Tensor.sum
    Tensor.swapaxes
    Tensor.swapdims
    Tensor.sub
    Tensor.sub_
    Tensor.tan
    Tensor.tanh
    Tensor.tile
    Tensor.to
    Tensor.local_to_global
    Tensor.global_to_global
    Tensor.to_global
    Tensor.to_local
    Tensor.to_consistent
    Tensor.tolist
    Tensor.topk
    Tensor.transpose
    Tensor.tril
    Tensor.triu
    Tensor.trunc
    Tensor.type_as
    Tensor.type
    Tensor.t
    Tensor.T
    Tensor.unbind
    Tensor.unfold
    Tensor.uniform_
    Tensor.unsqueeze
    Tensor.unsqueeze_
    Tensor.as_strided
    Tensor.as_strided_
    Tensor.var
    Tensor.view
    Tensor.view_as
    Tensor.where
    Tensor.zero_
    Tensor.nms
    Tensor.pin_memory
    Tensor.is_pinned
    Tensor.inverse
    Tensor.cross
    Tensor.scatter
    Tensor.scatter_
    Tensor.scatter_add
    Tensor.scatter_add_
    Tensor.bernoulli
    Tensor.bernoulli_
    Tensor.bincount
    Tensor.isclose
    Tensor.allclose
    Tensor.broadcast_to
    Tensor.unique
    Tensor.bitwise_and
    Tensor.bitwise_or
    Tensor.bitwise_xor
    Tensor.baddbmm
