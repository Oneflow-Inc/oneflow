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

.. currentmodule:: oneflow.nn
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: 

    Parameter


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

nn.Module
----------------------------------
.. currentmodule:: oneflow.nn.Module

.. autosummary::
    :toctree: generated
    :nosignatures:

    add_module
    apply
    buffers
    children
    cpu
    cuda
    double
    train
    eval
    extra_repr
    float
    forward
    load_state_dict
    modules
    named_buffers
    named_children
    named_modules
    named_parameters
    parameters
    register_buffer
    register_forward_hook
    register_forward_pre_hook
    register_backward_hook
    register_full_backward_hook
    register_state_dict_pre_hook
    register_parameter
    requires_grad_
    state_dict
    to
    zero_grad



Containers

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
    nn.MaxUnpool1d
    nn.MaxUnpool2d
    nn.MaxUnpool3d
    nn.AdaptiveAvgPool1d 
    nn.AdaptiveAvgPool2d 
    nn.AdaptiveAvgPool3d
    nn.AdaptiveMaxPool1d 
    nn.AdaptiveMaxPool2d 
    nn.AdaptiveMaxPool3d
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
    nn.ReflectionPad1d
    nn.ReflectionPad2d
    nn.ReplicationPad1d
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
    nn.QuickGELU 
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
    nn.SyncBatchNorm
    nn.FusedBatchNorm1d 
    nn.FusedBatchNorm2d
    nn.FusedBatchNorm3d 
    nn.GroupNorm 
    nn.InstanceNorm1d 
    nn.InstanceNorm2d 
    nn.InstanceNorm3d 
    nn.LayerNorm
    nn.RMSLayerNorm
    nn.RMSNorm

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
    :template: classtemplate.rst

    nn.Dropout
    nn.Dropout1d
    nn.Dropout2d
    nn.Dropout3d

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
    nn.PairwiseDistance

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


Data loading and preprocessing Layers
----------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.COCOReader
    nn.CoinFlip
    nn.CropMirrorNormalize
    nn.OFRecordBytesDecoder
    nn.OFRecordImageDecoder
    nn.OFRecordImageDecoderRandomCrop
    nn.OFRecordRawDecoder
    nn.OFRecordReader

Quantization Aware Training
--------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.MinMaxObserver
    nn.MovingAverageMinMaxObserver
    nn.FakeQuantization
    nn.QatConv1d
    nn.QatConv2d
    nn.QatConv3d

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

Quantized Functions
--------------------

Quantization refers to techniques for performing computations and 
storing tensors at lower bitwidths than floating point precision.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template:

    nn.FakeQuantization
    nn.MinMaxObserver
    nn.MovingAverageMinMaxObserver
    nn.Quantization
