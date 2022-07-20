oneflow.nn.functional
===========================================

.. The documentation is referenced from: https://pytorch.org/docs/1.10/nn.functional.html.

.. contents:: oneflow.nn.functional
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top

.. currentmodule:: oneflow.nn.functional

Convolution functions
-------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    conv1d
    conv2d
    conv3d
    conv_transpose1d
    conv_transpose2d
    conv_transpose3d
    fold
    unfold

Pooling functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    avg_pool1d
    avg_pool2d
    avg_pool3d
    max_pool1d
    max_pool2d
    max_pool3d
    adaptive_avg_pool1d
    adaptive_avg_pool2d
    adaptive_avg_pool3d

Non-linear activation functions
-------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    threshold
    relu
    hardtanh
    hardswish
    relu6
    elu
    selu
    celu
    leaky_relu
    prelu
    glu
    gelu
    logsigmoid
    hardshrink
    softsign
    softplus
    softmax
    softshrink
    log_softmax
    tanh
    sigmoid
    hardsigmoid
    silu
    mish
    layer_norm
    normalize

Linear functions
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    linear

Dropout functions
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dropout

Sparse functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    embedding
    one_hot

Distance functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    cosine_similarity


Loss functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse_softmax_cross_entropy
    cross_entropy
    smooth_l1_loss
    triplet_margin_loss

Vision functions
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    pad
    interpolate
    grid_sample
    affine_grid
