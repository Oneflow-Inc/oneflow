oneflow.nn
===================================

.. currentmodule:: oneflow.nn

Containers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: 

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

Convolution Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Conv1d 
    Conv2d 
    Conv3d
    ConvTranspose1d 
    ConvTranspose2d 
    ConvTranspose3d

Pooling Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    MaxPool1d 
    MaxPool2d 
    MaxPool3d 
    AdaptiveAvgPool1d 
    AdaptiveAvgPool2d 
    AdaptiveAvgPool3d
    AvgPool1d 
    AvgPool2d 
    AvgPool3d

Padding Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ConstantPad1d 
    ConstantPad2d 
    ConstantPad3d
    ZeroPad2d

Non-linear Activations (weighted sum, nonlinearity)
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    ELU 
    Hardsigmoid 
    Hardswish 
    Hardtanh 
    LeakyReLU 
    LogSigmoid 
    PReLU 
    ReLU 
    SELU 
    CELU 
    GELU 
    SiLU 
    Sigmoid 
    Mish 
    Softplus 
    Softshrink 
    Softsign 
    Tanh 
    Threshold 
    GLU

Non-linear Activations (other)
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Softmax
    LogSoftmax

Normalization Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BatchNorm1d 
    BatchNorm2d 
    BatchNorm3d 
    FusedBatchNorm1d 
    FusedBatchNorm2d
    FusedBatchNorm3d 
    GroupNorm 
    InstanceNorm1d 
    InstanceNorm2d 
    InstanceNorm3d 
    LayerNorm

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    Identity
    Linear

Dropout Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Dropout

Sparse Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Embedding

Loss Functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BCELoss 
    BCEWithLogitsLoss 
    CTCLoss 
    CombinedMarginLoss 
    CrossEntropyLoss 
    KLDivLoss 
    L1Loss 
    MSELoss 
    MarginRankingLoss 
    NLLLoss 
    SmoothL1Loss 
    TripletMarginLoss

Vision Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    PixelShuffle 
    Upsample 
    UpsamplingBilinear2d 
    UpsamplingNearest2d

DataParallel Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    DistributedDataParallel

DataParallel Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    
    utils.clip_grad_norm_ 
    utils.weight_norm 
    utils.remove_weight_norm

.. autofunction:: oneflow.nn.modules.pixelshuffle.PixelShufflev2

.. autofunction:: oneflow.nn.parallel.DistributedDataParallel

.. currentmodule:: oneflow.nn.utils
.. autofunction:: oneflow.nn.utils.clip_grad_norm_
.. autofunction:: oneflow.nn.utils.weight_norm
.. autofunction:: oneflow.nn.utils.remove_weight_norm
