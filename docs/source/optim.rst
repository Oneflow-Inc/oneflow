oneflow.optim
===================================
oneflow.optim is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can be also easily integrated in the future.

How to use a optimizer
----------------------------------
you have to construct an optimizer object, that will hold the current state and will update the parameters based on the computed gradients.

.. currentmodule:: oneflow.optim

Base class
----------

.. autoclass:: Optimizer

.. autosummary::
    :toctree: generated
    :nosignatures:

    Optimizer.add_param_group
    Optimizer.load_state_dict
    Optimizer.state_dict
    Optimizer.step
    Optimizer.zero_grad

Algorithms
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Adagrad
    Adam
    AdamW
    LAMB
    RMSprop

Adjust Learning Rate
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    lr_scheduler.CosineAnnealingLR
    lr_scheduler.CosineDecayLR 
    lr_scheduler.ExponentialLR 
    lr_scheduler.LambdaLR 
    lr_scheduler.MultiStepLR
    lr_scheduler.PolynomialLR 
    lr_scheduler.ReduceLROnPlateau 
    lr_scheduler.StepLR