oneflow
===================================

.. The documentation is referenced from: 
   https://pytorch.org/docs/1.10/torch.html

The oneflow package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute capability >= 3.0

.. currentmodule:: oneflow


Tensor
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    is_tensor
    is_floating_point
    is_nonzero
    numel
    set_printoptions

.. _tensor-creation-ops:

Creation Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    Random sampling creation ops are listed under :ref:`random-sampling` and
    include:
    :func:`oneflow.rand`
    :func:`oneflow.randn`
    :func:`oneflow.randint`
    :func:`oneflow.randperm`
    
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    as_tensor
    as_strided
    from_numpy
    zeros
    zeros_like
    ones
    ones_like
    arange
    linspace
    eye
    empty
    full
    full_like

.. _indexing-slicing-joining:

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    argwhere
    cat
    concat
    chunk
    gather
    hsplit
    vsplit
    index_select
    masked_select
    movedim
    narrow
    nonzero
    permute
    reshape
    select
    scatter
    scatter_add
    scatter_nd
    split
    squeeze
    stack
    swapaxes
    swapdims
    t
    tile
    transpose
    unbind
    unsqueeze
    where
    tensor_split

.. _random-sampling:

Random sampling
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    seed
    manual_seed
    initial_seed
    get_rng_state
    set_rng_state
    bernoulli
    normal
    rand
    randint
    randn
    randperm
    
In-place random sampling
~~~~~~~~~~~~~~~~~~~~~~~~

There are a few more in-place random sampling functions defined on Tensors as well. Click through to refer to their documentation:
- :func:`oneflow.Tensor.normal_` - in-place version of :func:`oneflow.normal`
- :func:`oneflow.Tensor.uniform_` - numbers sampled from the continuous uniform distribution



Serialization
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    save
    load

Parallelism
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    set_num_threads


Locally disabling gradient computation
-------------------------------------------
The context managers :func:`oneflow.no_grad`, :func:`oneflow.enable_grad`, and
:func:`oneflow.set_grad_enabled` are helpful for locally disabling and enabling
gradient computation. See :ref:`locally-disable-grad` for more details on
their usage.  These context managers are thread local, so they won't
work if you send work to another thread using the ``threading`` module, etc.

Examples::

  >>> import oneflow
  >>> x = oneflow.zeros(1, requires_grad=True)
  >>> with oneflow.no_grad():
  ...     y = x * 2
  >>> y.requires_grad
  False

  >>> with oneflow.set_grad_enabled(False):
  ...     y = x * 2
  >>> y.requires_grad
  False
  
  >>> with oneflow.set_grad_enabled(True):
  ...     y = x * 2
  >>> y.requires_grad
  True

.. autosummary::
    :toctree: generated
    :nosignatures:

    no_grad
    set_grad_enabled
    enable_grad
    is_grad_enabled
    inference_mode

Math operations
-------------------------------------------

Pointwise Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs 
    acos 
    acosh 
    arccos 
    arccosh
    add 
    addcmul
    asin 
    asinh 
    arcsin 
    arcsinh 
    atan
    atanh 
    arctan 
    arctanh 
    atan2 
    ceil 
    clamp 
    clip 
    cos 
    cosh 
    div 
    erf 
    erfc 
    erfinv
    exp 
    expm1 
    floor 
    floor_ 
    fmod 
    log 
    log1p 
    log2 
    logical_and 
    logical_not 
    logical_or 
     
    logical_xor 
    mul 
    neg 
    negative 
    pow 
    reciprocal 
    round 
    rsqrt 
    sigmoid 
    sign 
    sin 
    sinh 
    sin_ 
    sqrt 
    square 
    sub 
    tan 
    tanh
    floor_divide

Reduction Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    argmax  
    argmin  
    amax
    amin
    any
    max
    min  
    mean  
    median
    prod
    std  
    sum  
    var
    norm
    all


Comparison Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    argsort 
    eq 
    equal 
    gt 
    isinf 
    isnan 
    le 
    lt 
    ne 
    sort 
    topk
    ge
    greater
    greater_equal
    maximum
    minimum
    not_equal

Other Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    
    broadcast_like 
    cumprod 
    cumsum 
    diag 
    diagonal 
    einsum 
    flatten 
    flip 
    meshgrid 
    roll 
    searchsorted
    tensordot
    tril
    repeat_interleave
    triu

BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    addmm 
    bmm 
    dot 
    matmul
    mm
    mv



