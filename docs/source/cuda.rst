oneflow.cuda
===================================

.. The documentation is referenced from: https://pytorch.org/docs/1.10/cuda.html.

.. currentmodule:: oneflow.cuda

.. autosummary::
    :toctree: generated
    :nosignatures:

    is_available
    device_count
    current_device
    set_device
    synchronize

.. note::
   The :attr:`current_device` returns local rank as device index. It is different from the 'torch.current_device()' in PyTorch.


Random Number Generator
-------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    manual_seed_all
    manual_seed


GPU tensor
-----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    HalfTensor
    FloatTensor
    DoubleTensor
    BoolTensor
    ByteTensor
    CharTensor
    IntTensor
    LongTensor

Memory management
-----------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    empty_cache