oneflow.autograd
================================================
oneflow.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword. 

.. currentmodule:: oneflow.autograd

Functions and classes for autograd.
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    grad
    backward

Tensor autograd functions
---------------------------------------------------
.. autosummary::
    :nosignatures:

   oneflow.Tensor.grad
   oneflow.Tensor.requires_grad
   oneflow.Tensor.is_leaf
   oneflow.Tensor.backward
   oneflow.Tensor.detach
   oneflow.Tensor.register_hook
   oneflow.Tensor.retain_grad

.. autoclass:: oneflow.autograd.Function
    :members: apply,
    :special-members: __call__,
