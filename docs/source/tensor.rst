oneflow.Tensor
===================================

A :class:`oneflow.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

.. currentmodule:: oneflow

Data types
----------

OneFlow defines 8 tensor types with CPU and GPU variants which are as follows:

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

    >>> oneflow.tensor([[1., -1.], [1., -1.]])
    tensor([[ 1.0000, -1.0000],
            [ 1.0000, -1.0000]])
    >>> oneflow.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    tensor([[ 1,  2,  3],
            [ 4,  5,  6]])

.. warning::

    :func:`oneflow.tensor` always copies :attr:`data`. If you have a Tensor
    :attr:`data` and just want to change its ``requires_grad`` flag, use
    :meth:`~oneflow.Tensor.requires_grad_` or
    :meth:`~oneflow.Tensor.detach` to avoid a copy.
    If you have a numpy array and want to avoid a copy, use
    :func:`oneflow.as_tensor`.
A tensor of specific data type can be constructed by passing a
:class:`oneflow.dtype` and/or a :class:`oneflow.device` to a
constructor or tensor creation op:

::

    >>> oneflow.zeros([2, 4], dtype=oneflow.int32)
    tensor([[ 0,  0,  0,  0],
            [ 0,  0,  0,  0]], dtype=oneflow.int32)
    >>> cuda0 = oneflow.device('cuda:0')
    >>> oneflow.ones([2, 4], dtype=oneflow.float64, device=cuda0)
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.0000,  1.0000,  1.0000,  1.0000]], dtype=oneflow.float64, device='cuda:0')

For more information about building Tensors, see :ref:`tensor-creation-ops`

The contents of a tensor can be accessed and modified using Python's indexing
and slicing notation:

::

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
    :meth:`~oneflow.Tensor.to` method on the tensor.

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

.. autoattribute:: Tensor.T

.. autoclass:: oneflow.Tensor
    :members: abs, 
            acos, 
            acosh, 
            add, 
            add_, 
            addcmul,
            addcmul_,
            addmm,
            all,
            amin,
            amax,
            any,
            arccos, 
            arccosh, 
            arcsin, 
            arcsinh, 
            arctan, 
            arctanh, 
            argmax, 
            argmin, 
            argsort, 
            argwhere, 
            asin, 
            asinh, 
            atan, 
            atan2, 
            atanh, 
            backward,
            bmm, 
            byte, 
            cast, 
            ceil, 
            chunk,  
            clamp, 
            clamp_,
            clip, 
            clip_, 
            clone, 
            contiguous,
            copy_, 
            cos, 
            cosh, 
            cpu, 
            cuda,
            cumprod,
            cumsum,
            data, 
            dot,
            detach, 
            device, 
            placement,
            sbp,
            diag, 
            diagonal,
            dim, 
            div, 
            div_, 
            double, 
            dtype, 
            element_size, 
            eq, 
            erf, 
            erfc, 
            erfinv, 
            erfinv_, 
            exp, 
            expand, 
            expand_as, 
            expm1, 
            fill_, 
            flatten, 
            flip, 
            float, 
            floor, 
            floor_, 
            floor_divide, 
            fmod,
            gather, 
            ge, 
            gelu, 
            get_device, 
            grad, 
            grad_fn, 
            gt, 
            half,
            in_top_k, 
            index_select,
            int, 
            is_global, 
            is_contiguous, 
            is_cuda, 
            is_floating_point, 
            is_lazy, 
            is_leaf, 
            isinf, 
            isnan, 
            item, 
            le, 
            log, 
            log1p,
            log2, 
            logical_and,
            logical_or,
            logical_not,
            logical_xor,
            long, 
            lt, 
            masked_fill, 
            masked_select, 
            matmul, 
            mm, 
            mv, 
            max, 
            maximum, 
            median, 
            mean, 
            min, 
            minimum, 
            mish, 
            mul, 
            mul_, 
            narrow, 
            ndim, 
            ndimension, 
            ne, 
            neg, 
            negative, 
            nelement, 
            new_empty,
            new_ones, 
            new_zeros,
            new_tensor, 
            nonzero,
            norm, 
            normal_, 
            numel, 
            numpy, 
            permute, 
            pow, 
            prod,
            reciprocal, 
            register_hook, 
            relu,
            repeat,
            repeat_interleave,
            requires_grad,
            requires_grad_,
            reshape, 
            reshape_as, 
            retain_grad,
            roll,
            round, 
            rsqrt, 
            selu, 
            shape, 
            sigmoid, 
            sign, 
            silu, 
            sin, 
            sin_, 
            sinh, 
            size, 
            softmax, 
            softplus, 
            softsign, 
            sort, 
            split, 
            sqrt, 
            square, 
            squeeze, 
            std, 
            storage_offset, 
            stride, 
            sum,
            swapaxes, 
            swapdims, 
            sub, 
            sub_, 
            tan, 
            tanh, 
            tile, 
            to,
            local_to_global,
            global_to_global,
            to_global,
            to_local,
            to_consistent,
            tolist, 
            topk, 
            transpose,
            tril, 
            triu, 
            type_as, 
            type,
            t,
            T,
            unbind, 
            unfold, 
            uniform_, 
            unsqueeze, 
            var, 
            view, 
            view_as, 
            where, 
            zero_, 
            nms,
            pin_memory,
            is_pinned,

