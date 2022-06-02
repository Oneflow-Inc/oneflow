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