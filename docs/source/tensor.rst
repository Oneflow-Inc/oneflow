oneflow.Tensor
===================================

A :class:`oneflow.Tensor` is a multi-dimensional matrix containing elements of
a single data type.

.. currentmodule:: oneflow

Data types
----------

Torch defines 10 tensor types with CPU and GPU variants which are as follows:

======================================= =============================================== =============================== ==================================
Data type                               dtype                                           CPU tensor                      GPU tensor
======================================= =============================================== =============================== ==================================
32-bit floating point                   ``oneflow.float32`` or ``oneflow.float``        :class:`oneflow.FloatTensor`    :class:`oneflow.cuda.FloatTensor`
64-bit floating point                   ``oneflow.float64`` or ``oneflow.double``       :class:`oneflow.DoubleTensor`   :class:`oneflow.cuda.DoubleTensor`
16-bit floating point                   ``oneflow.float16`` or ``oneflow.half``         :class:`oneflow.HalfTensor`     :class:`oneflow.cuda.HalfTensor`
32-bit complex                          ``oneflow.complex32`` or ``oneflow.chalf``
64-bit complex                          ``oneflow.complex64`` or ``oneflow.cfloat``
128-bit complex                         ``oneflow.complex128`` or ``oneflow.cdouble``
8-bit integer (unsigned)                ``oneflow.uint8``                               :class:`oneflow.ByteTensor`     :class:`oneflow.cuda.ByteTensor`
8-bit integer (signed)                  ``oneflow.int8``                                :class:`oneflow.CharTensor`     :class:`oneflow.cuda.CharTensor`
32-bit integer (signed)                 ``oneflow.int32`` or ``oneflow.int``            :class:`oneflow.IntTensor`      :class:`oneflow.cuda.IntTensor`
64-bit integer (signed)                 ``oneflow.int64`` or ``oneflow.long``           :class:`oneflow.LongTensor`     :class:`oneflow.cuda.LongTensor`
Boolean                                 ``oneflow.bool``                                :class:`oneflow.BoolTensor`     :class:`oneflow.cuda.BoolTensor`
quantized 8-bit integer (unsigned)      ``oneflow.quint8``                              :class:`oneflow.ByteTensor`     /
quantized 8-bit integer (signed)        ``oneflow.qint8``                               :class:`oneflow.CharTensor`     /
quantized 32-bit integer (signed)       ``oneflow.qint32``                              :class:`oneflow.IntTensor`      /
quantized 4-bit integer (unsigned)      ``oneflow.quint4x2``                            :class:`oneflow.ByteTensor`     /
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

.. autoclass:: oneflow.Tensor
    :members: abs, 
            acos, 
            acosh, 
            add, 
            add_, 
            addcmul,
            addcmul_,
            addmm,
            amin,
            amax,
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
            copy_, 
            cos, 
            cosh, 
            cpu, 
            cuda,
            data, 
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
            item, 
            le, 
            log, 
            log1p,
            logical_and,
            logical_or,
            logical_not,
            logical_xor,
            long, 
            lt, 
            masked_fill, 
            masked_select, 
            matmul, 
            max, 
            mean, 
            min, 
            mish, 
            mul, 
            mul_, 
            narrow, 
            ndim, 
            ndimension, 
            ne, 
            negative, 
            nelement, 
            new_empty,
            new_ones, 
            new_zeros,
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
            requires_grad, 
            requires_grad_,
            reshape, 
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