.. role:: hidden
    :class: hidden-section

oneflow
===================================
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

Creation Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
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

Indexing, Slicing, Joining, Mutating Ops
-------------------------------------------

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
    oneflow.scatter_nd
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


Random sampling
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    seed
    manual_seed
    initial_seed
    get_rng_state
    bernoulli
    rand
    randint
    randn
    randperm
    set_rng_state

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
    add 
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
    erfin 
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
    logical_slice 
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

Reduction Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    argmax  
    argmin  
    min  
    mean  
    prod
    prod
    std  
    sum  
    var


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
    tril

BLAS and LAPACK Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    addmm 
    bmm 
    dot 
    matmul


