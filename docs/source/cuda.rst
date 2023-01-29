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
    get_device_properties
    get_device_capability
    get_device_name

.. note::
   The :attr:`current_device` returns local rank as device index. It is different from the 'torch.current_device()' in PyTorch.


Random Number Generator
-------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    manual_seed_all
    manual_seed
    get_rng_state
    get_rng_state_all


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
    