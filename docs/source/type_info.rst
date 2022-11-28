.. currentmodule:: oneflow

.. _type-info-doc:

Type Info
=========

.. The documentation is referenced from: https://pytorch.org/docs/1.10/type_info.html.

The numerical properties of a :class:`oneflow.dtype` can be accessed through either the :class:`oneflow.finfo` or the :class:`oneflow.iinfo`.


.. contents:: oneflow
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top

oneflow.finfo
-------------

.. class:: oneflow.finfo

A :class:`oneflow.finfo` is an object that represents the numerical properties of a floating point :class:`oneflow.dtype`, (i.e. ``oneflow.float32``, ``oneflow.float64`` and ``oneflow.float16``). This is similar to `numpy.finfo <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

A :class:`oneflow.finfo` provides the following attributes:

================== ======= ========================================================================== 
Name               Type    Description                                                               
================== ======= ========================================================================== 
bits               int     The number of bits occupied by the type.                                  
eps                float   The smallest representable number such that ``1.0 + eps != 1.0``.             
min                float   The largest representable number.                                         
max                float   The smallest representable number (typically ``-max``).                       
tiny               float   The smallest positive normal number. See notes.
resolution         float   The approximate decimal resolution of this type, i.e., ``10**-precision``.    
================== ======= ========================================================================== 

For example:

.. code-block::

    >>> import oneflow as flow
    >>> flow.finfo()
    finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, tiny=1.17549e-38, dtype=oneflow.float32, bits=32)
    >>> flow.finfo(flow.float)
    finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, tiny=1.17549e-38, dtype=oneflow.float32, bits=32)
    >>> flow.finfo(flow.float16).bits
    16
    >>> flow.finfo(flow.float16).max
    65504.0

oneflow.iinfo
-------------

.. class:: oneflow.iinfo

A :class:`oneflow.iinfo` is an object that represents the numerical properties of a integer :class:`oneflow.dtype` (i.e. ``oneflow.uint8``, ``oneflow.int8``, ``oneflow.int16``, ``oneflow.int32``, and ``oneflow.int64``). This is similar to `numpy.iinfo <https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html>`_.

A :class:`oneflow.iinfo` provides the following attributes:

================== ======= ========================================================================== 
Name               Type    Description                                                               
================== ======= ========================================================================== 
bits               int     The number of bits occupied by the type.                                  
min                float   The largest representable number.                                         
max                float   The smallest representable number.                       
================== ======= ========================================================================== 

For example:

.. code-block ::

    >>> import oneflow as flow
    >>> flow.iinfo(flow.int8)
    iinfo(min=-128, max=127, dtype=oneflow.int8, bits=8)
    >>> flow.iinfo(flow.int).max
    2147483647
    >>> flow.iinfo(flow.int).bits
    32
