oneflow.autograd
====================================================

.. The documentation is referenced from:
   https://pytorch.org/docs/1.10/autograd.html

``oneflow.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar 
valued functions. It requires minimal changes to the existing code - you only need to declare ``Tensor`` s 
for which gradients should be computed with the ``requires_grad=True`` keyword. As of now, we only support 
autograd for floating point ``Tensor`` types ( half, float, double and bfloat16).


.. currentmodule:: oneflow.autograd

.. autosummary::
    :toctree: generated
    :nosignatures:

    backward
    grad

Locally disabling gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:

    no_grad
    enable_grad
    set_grad_enabled
    inference_mode

.. TODO(wyg): uncomment this after aligning accumulate grad
.. Default gradient layouts
.. ^^^^^^^^^^^^^^^^^^^^^^^^

.. A ``param.grad`` is accumulated by replacing ``.grad`` with a 
.. new tensor ``.grad + new grad`` during :func:`oneflow.autograd.backward()` or 
.. :func:`oneflow.Tensor.backward()`.

In-place operations on Tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

Tensor autograd functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :nosignatures:

   oneflow.Tensor.grad
   oneflow.Tensor.requires_grad
   oneflow.Tensor.is_leaf
   oneflow.Tensor.backward
   oneflow.Tensor.detach
   oneflow.Tensor.register_hook
   oneflow.Tensor.retain_grad

Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Function
.. currentmodule:: oneflow.autograd
.. autosummary::
    :toctree: generated
    :nosignatures:

    Function.forward
    Function.backward
    Function.apply

Context method mixins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When creating a new :class:`Function`, the following methods are available to `ctx`.

.. currentmodule:: oneflow._oneflow_internal.autograd.Function
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    FunctionCtx.mark_non_differentiable
    FunctionCtx.save_for_backward
    FunctionCtx.saved_tensors
