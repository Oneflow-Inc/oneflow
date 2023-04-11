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

BatchNorm functions
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    batch_norm

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
    max_unpool1d
    max_unpool2d
    max_unpool3d
    adaptive_avg_pool1d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    adaptive_max_pool1d
    adaptive_max_pool2d
    adaptive_max_pool3d

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
    quick_gelu
    logsigmoid
    hardshrink
    softsign
    softplus
    softmax
    softshrink
    log_softmax
    gumbel_softmax
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
    dropout1d
    dropout2d
    dropout3d

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
    pairwise_distance


Loss functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse_softmax_cross_entropy
    cross_entropy
    ctc_loss
    l1_loss
    mse_loss
    smooth_l1_loss
    triplet_margin_loss
    binary_cross_entropy
    binary_cross_entropy_with_logits

Vision functions
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    deform_conv2d
    pad
    interpolate
    upsample
    grid_sample
    affine_grid

Greedy decoder
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ctc_greedy_decoder

