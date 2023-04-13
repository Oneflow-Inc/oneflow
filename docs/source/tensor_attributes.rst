.. currentmodule:: oneflow

.. _tensor-attributes-doc:

Tensor Attributes
=============================================================

.. The documentation is referenced from: https://pytorch.org/docs/1.10/tensor_attributes.html.


Each local ``oneflow.Tensor`` has a :class:`oneflow.dtype`, :class:`oneflow.device`, and global ``oneflow.Tensor`` has a :class:`oneflow.dtype`, :class:`oneflow.placement`, :class:`oneflow.sbp`.

.. contents:: oneflow
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top


.. _dtype-doc:

oneflow.dtype
-----------------------

.. class:: dtype

A :class:`oneflow.dtype` is an object that represents the data type of a
:class:`oneflow.Tensor`. Oneflow has eight different data types:

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


To find out if a :class:`oneflow.dtype` is a floating point data type, the property :attr:`is_floating_point`
can be used, which returns ``True`` if the data type is a floating point data type.

.. _type-promotion-doc:

When the dtypes of inputs to an arithmetic operation (`add`, `sub`, `div`, `mul`) differ, we promote
by finding the minimum dtype that satisfies the following rules:

* If the type of a scalar operand is of a higher category than tensor operands
  (where complex > floating > integral > boolean), we promote to a type with sufficient size to hold
  all scalar operands of that category.
* If a zero-dimension tensor operand has a higher category than dimensioned operands,
  we promote to a type with sufficient size and category to hold all zero-dim tensor operands of
  that category.
* If there are no higher-category zero-dim operands, we promote to a type with sufficient size
  and category to hold all dimensioned operands.

A floating point scalar operand has dtype `oneflow.get_default_dtype()` and an integral
non-boolean scalar operand has dtype `oneflow.int64`. Unlike numpy, we do not inspect
values when determining the minimum `dtypes` of an operand.  Quantized and complex types
are not yet supported.

Promotion Examples::

    >>> float_tensor = oneflow.ones(1, dtype=oneflow.float)
    >>> double_tensor = oneflow.ones(1, dtype=oneflow.double)
    >>> int_tensor = oneflow.ones(1, dtype=oneflow.int)
    >>> long_tensor = oneflow.ones(1, dtype=oneflow.long)
    >>> uint_tensor = oneflow.ones(1, dtype=oneflow.uint8)
    >>> double_tensor = oneflow.ones(1, dtype=oneflow.double)
    >>> bool_tensor = oneflow.ones(1, dtype=oneflow.bool)
    # zero-dim tensors
    >>> long_zerodim = oneflow.tensor(1, dtype=oneflow.long)
    >>> int_zerodim = oneflow.tensor(1, dtype=oneflow.int)

    >>> a,b=oneflow.tensor(5),oneflow.tensor(5)
    >>> oneflow.add(a, b).dtype
    oneflow.int64
    # 5 is an int64, but does not have higher category than int_tensor so is not considered.
    >>> (int_tensor + 5).dtype
    oneflow.int32
    >>> (int_tensor + long_zerodim).dtype
    oneflow.int64
    >>> (long_tensor + int_tensor).dtype
    oneflow.int64
    >>> (bool_tensor + long_tensor).dtype
    oneflow.int64
    >>> (bool_tensor + uint_tensor).dtype
    oneflow.uint8
    >>> (float_tensor + double_tensor).dtype
    oneflow.float64
    >>> (bool_tensor + int_tensor).dtype
    oneflow.int32
    # Since long is a different kind than float, result dtype only needs to be large enough
    # to hold the float.
    >>> oneflow.add(long_tensor, float_tensor).dtype
    oneflow.float32

When the output tensor of an arithmetic operation is specified, we allow casting to its `dtype` except that:
  * An integral output tensor cannot accept a floating point tensor.
  * A boolean output tensor cannot accept a non-boolean tensor.
  * A non-complex output tensor cannot accept a complex tensor

Casting Examples::

    # allowed:
    >>> float_tensor *= float_tensor
    >>> float_tensor *= int_tensor
    >>> float_tensor *= uint_tensor
    >>> float_tensor *= bool_tensor
    >>> int_tensor *= uint_tensor

    # disallowed (RuntimeError: result type can't be cast to the desired output type):
    >>> float_tensor *= double_tensor
    >>> int_tensor *= float_tensor
    >>> int_tensor *= long_tensor
    >>> uint_tensor *= int_tensor
    >>> bool_tensor *= int_tensor
    >>> bool_tensor *= uint_tensor

.. _device-doc:

oneflow.device
------------------------

.. class:: device

A :class:`oneflow.device` is an object representing the device on which a :class:`oneflow.Tensor` is
or will be allocated.

The :class:`oneflow.device` contains a device type (``'cpu'`` or ``'cuda'``) and optional device
ordinal for the device type. If the device ordinal is not present, this object will always represent
the current device for the device type, even after :func:`oneflow.cuda.set_device()` is called; e.g.,
a :class:`oneflow.Tensor` constructed with device ``'cuda'`` is equivalent to ``'cuda:X'`` where X is
the result of :func:`oneflow.cuda.current_device()`.

A :class:`oneflow.Tensor`'s device can be accessed via the :attr:`Tensor.device` property.

A :class:`oneflow.device` can be constructed via a string or via a string and device ordinal

Via a string:
::

    >>> oneflow.device('cuda:0')
    device(type='cuda', index=0)

    >>> oneflow.device('cpu')
    device(type='cpu', index=0)

    >>> oneflow.device('cuda')  # current cuda device
    device(type='cuda', index=0)

Via a string and device ordinal:

::

    >>> oneflow.device('cuda', 0)
    device(type='cuda', index=0)

    >>> oneflow.device('cpu', 0)
    device(type='cpu', index=0)

.. note::
   The :class:`oneflow.device` argument in functions can generally be substituted with a string.
   This allows for fast prototyping of code.

   >>> # Example of a function that takes in a oneflow.device
   >>> cuda1 = oneflow.device('cuda:1')
   >>> oneflow.randn((2,3), device=cuda1)

   >>> # You can substitute the oneflow.device with a string
   >>> oneflow.randn((2,3), device='cuda:1')

.. note::
   For legacy reasons, a device can be constructed via a single device ordinal, which is treated
   as a cuda device.  This matches :meth:`Tensor.get_device`, which returns an ordinal for cuda
   tensors and is not supported for cpu tensors.

   >>> oneflow.device(1)
   device(type='cuda', index=1)

.. note::
   Methods which take a device will generally accept a (properly formatted) string
   or (legacy) integer device ordinal, i.e. the following are all equivalent:

   >>> oneflow.randn((2,3), device=oneflow.device('cuda:1'))
   >>> oneflow.randn((2,3), device='cuda:1')
   >>> oneflow.randn((2,3), device=1)  # legacy

oneflow.placement
--------------------------------------------------------------
.. autoclass:: oneflow.placement

oneflow.placement.all
--------------------------------------------------------------
.. autofunction:: oneflow.placement.all

oneflow.env.all_device_placement
--------------------------------------------------------------
.. autofunction:: oneflow.env.all_device_placement

oneflow.sbp.sbp
--------------------------------------------------------------
.. autoclass:: oneflow.sbp.sbp
