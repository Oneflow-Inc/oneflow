.. role:: hidden
    :class: hidden-section

oneflow
===================================

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
-------------------------------------------

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
    set_rng_state

oneflow.default_generator Returns the default CPU oneflow.Generator
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    bernoulli
    normal
    rand
    randn
    randint
    randperm

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

    oneflow.set_num_threads

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


.. automodule:: oneflow
    :members: adaptive_avg_pool1d, 
            adaptive_avg_pool2d, 
            adaptive_avg_pool3d, 
            abs, 
            acos, 
            acosh, 
            add, 
            addcmul, 
            addmm, 
            all, 
            amin, 
            amax,
            any, 
            arccos, 
            arcsin, 
            arcsinh, 
            arccosh, 
            arctan, 
            arctanh, 
            argmax, 
            argmin, 
            
            argsort, 
            
            asin,  
            asinh, 
            atan, 
            atan2, 
            atanh, 
            
            broadcast_like, 
            batch_gather,
            bmm,
            
            cast, 
            ceil, 
            
            clamp, 
            clip, 
            cos, 
            cosh, 
            diag, 
            
            diagonal,
            
            tensor_split,

            
            div, 
            dot, 
            eq,
            einsum,
            equal, 
            expand, 
            
            exp, 
            expm1, 
            erf, 
            erfc, 
            erfinv, 
            flatten, 
            flip, 
            floor, 
            floor_,
            fmod,
            
             
            gather_nd, 
            gelu, 
            gt, 
            in_top_k, 
            
            
            logical_and,
            logical_or,
            logical_not,
            logical_xor,
            
            log, 
            log2,
            log1p, 
            lt, 
            le, 
            masked_fill, 
            
            matmul, 
             
            max, 
            mean,
            median,
            mish,  
            min, 
            meshgrid,
            mul, 
            neg, 
            negative, 
            new_ones,
            
            
            numel, 
            ne, 
            
            
            
            pow,
            prod,  

            repeat, 
             

            reciprocal,
            roc_auc_score,
            roll,
            round, 
            rsqrt,
     
            
            
            scatter_nd, 
            tensor_scatter_nd_update,
            sin, 
            sin_, 
            sinh, 
            sign, 
            selu, 
            silu, 
            slice, 
            logical_slice,  
            softsign, 
            sort, 
            softplus, 
            sigmoid, 
            softmax, 
           
            
            
            std,
            sub, 
            sum, 
            sqrt, 
            square,  
            
            
            tan, 
            tanh, 
            
            tensordot,
             
            
            
            tril, 
             
            
            
            var, 
            
            
            
            is_nonzero,
            is_tensor,

            
            is_floating_point,
            set_printoptions,
            decode_onerec,
            
            
            cumsum,
            topk,
            nms,
            cumprod,
            HalfTensor,
            FloatTensor,
            DoubleTensor,
            BoolTensor,
            ByteTensor,
            CharTensor,
            IntTensor,
            LongTensor,


            isnan,
            isinf,
            searchsorted

.. autofunction:: oneflow.relu


