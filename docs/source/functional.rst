.. role:: hidden
    :class: hidden-section

oneflow.nn.functional
===========================================

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

Pooling functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

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


Loss functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse_softmax_cross_entropy
    cross_entropy
    triplet_margin_loss

Vision functions
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    pad
    interpolate
    upsample
    grid_sample
    affine_grid