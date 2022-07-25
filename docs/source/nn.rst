oneflow.nn
===================================

.. The documentation is referenced from: 
   https://pytorch.org/docs/1.10/nn.html

These are the basic building blocks for graphs:

.. contents:: oneflow.nn
    :depth: 2
    :local:
    :class: this-will-duplicate-information-and-it-is-still-useful-here
    :backlinks: top


Containers
----------------------------------
.. currentmodule:: oneflow.nn

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

Convolution Layers
----------------------------------
.. currentmodule:: oneflow
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Conv1d 
    nn.Conv2d 
    nn.Conv3d
    nn.ConvTranspose1d 
    nn.ConvTranspose2d 
    nn.ConvTranspose3d
    nn.Unfold
    nn.Fold

Pooling Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MaxPool1d 
    nn.MaxPool2d 
    nn.MaxPool3d 
    nn.AdaptiveAvgPool1d 
    nn.AdaptiveAvgPool2d 
    nn.AdaptiveAvgPool3d
    nn.AvgPool1d 
    nn.AvgPool2d 
    nn.AvgPool3d

Padding Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ConstantPad1d 
    nn.ConstantPad2d 
    nn.ConstantPad3d
    nn.ReflectionPad2d
    nn.ReplicationPad2d
    nn.ZeroPad2d

Non-linear Activations (weighted sum, nonlinearity)
----------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ELU 
    nn.Hardshrink
    nn.Hardsigmoid 
    nn.Hardswish 
    nn.Hardtanh 
    nn.LeakyReLU 
    nn.LogSigmoid 
    nn.PReLU 
    nn.ReLU
    nn.ReLU6 
    nn.SELU 
    nn.CELU 
    nn.GELU 
    nn.SiLU 
    nn.Sigmoid 
    nn.Mish 
    nn.Softplus 
    nn.Softshrink 
    nn.Softsign 
    nn.Tanh 
    nn.Threshold 
    nn.GLU

Non-linear Activations (other)
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Softmax
    nn.LogSoftmax

Normalization Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BatchNorm1d 
    nn.BatchNorm2d 
    nn.BatchNorm3d 
    nn.FusedBatchNorm1d 
    nn.FusedBatchNorm2d
    nn.FusedBatchNorm3d 
    nn.GroupNorm 
    nn.InstanceNorm1d 
    nn.InstanceNorm2d 
    nn.InstanceNorm3d 
    nn.LayerNorm

Recurrent Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Identity
    nn.Linear

Dropout Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.Dropout

Sparse Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding

Distance Functions
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.CosineSimilarity

Loss Functions
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BCELoss 
    nn.BCEWithLogitsLoss 
    nn.CTCLoss 
    nn.CombinedMarginLoss 
    nn.CrossEntropyLoss 
    nn.KLDivLoss 
    nn.L1Loss 
    nn.MSELoss 
    nn.MarginRankingLoss 
    nn.NLLLoss 
    nn.SmoothL1Loss 
    nn.TripletMarginLoss

Vision Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PixelShuffle 
    nn.Upsample 
    nn.UpsamplingBilinear2d 
    nn.UpsamplingNearest2d


DataParallel Layers (multi-GPU, distributed)
--------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst
    
    nn.parallel.DistributedDataParallel

Utilities
---------

From the ``oneflow.nn.utils`` module

.. currentmodule:: oneflow.nn.utils
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    clip_grad_norm_
    clip_grad_value_
    weight_norm
    remove_weight_norm

Utility functions in other modules

.. currentmodule:: oneflow
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.utils.rnn.PackedSequence
    nn.utils.rnn.pack_padded_sequence
    nn.utils.rnn.pad_packed_sequence
    nn.utils.rnn.pad_sequence
    nn.utils.rnn.pack_sequence

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Flatten
