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

    BoolTensor
    ByteTensor
    CharTensor
    DoubleTensor
    FloatTensor
    HalfTensor
    IntTensor
    LongTensor


.. autosummary::
    :toctree: generated
    :nosignatures:

    is_tensor
    is_floating_point
    is_nonzero
    numel
    set_printoptions
    get_default_dtype
    set_default_dtype
    set_default_tensor_type

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
    randn_like
    randint_like
    masked_fill
    new_ones
    arange
    linspace
    eye
    empty
    empty_like
    full
    full_like
    tensor_scatter_nd_update
    logspace

.. _indexing-slicing-joining:

Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    argwhere
    atleast_1d
    atleast_2d
    atleast_3d
    cat
    column_stack
    concat
    chunk
    dstack
    expand
    gather
    gather_nd
    batch_gather
    hsplit
    hstack
    vsplit
    vstack
    index_select
    index_add
    masked_select
    movedim
    narrow
    nonzero
    permute
    repeat
    reshape
    row_stack
    select
    scatter
    scatter_add
    scatter_nd
    slice
    slice_update
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
    multinomial
    
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
gradient computation. These context managers are thread local, so they won't
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
    addcdiv
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
    ceil_
    clamp 
    clamp_min
    clamp_max
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
    frac
    frac_
    fmod 
    gelu
    quick_gelu
    log 
    log1p 
    log2 
    log10
    logical_and 
    logical_not 
    logical_or 
    logical_xor 
    bitwise_and
    bitwise_or
    bitwise_xor
    bitwise_not
    mish
    mul 
    neg 
    negative 
    pow 
    reciprocal 
    round 
    round_
    rsqrt 
    selu
    softmax
    softplus
    softsign
    silu
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
    trunc
    floor_divide
    lerp
    lerp_
    quantile

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
    mode
    prod
    nansum
    std  
    sum  
    logsumexp
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
    isclose
    allclose

Spectral Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    hann_window
    
Other Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    adaptive_avg_pool1d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    broadcast_like 
    cast
    cumprod 
    cumsum 
    diag 
    diagonal 
    einsum 
    flatten 
    flip 
    in_top_k
    meshgrid 
    nms
    roc_auc_score
    roll 
    searchsorted
    tensordot
    tril
    repeat_interleave
    triu
    cross
    bincount
    broadcast_shapes
    broadcast_tensors
    broadcast_to
    unique

BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    addmm 
    bmm
    baddbmm 
    dot 
    matmul
    mm
    mv

