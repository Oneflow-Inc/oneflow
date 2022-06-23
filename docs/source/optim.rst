oneflow.optim
===================================
oneflow.optim is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can be also easily integrated in the future.

How to use an optimizer
-----------------------

To use :mod:`oneflow.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

Constructing it
^^^^^^^^^^^^^^^

To construct an :class:`Optimizer` you have to give it an iterable containing the
parameters (all should be :class:`~oneflow.autograd.Variable` s) to optimize. Then,
you can specify optimizer-specific options such as the learning rate, weight decay, etc.

Example::

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

Per-parameter options
^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` s also support specifying per-parameter options. To do this, instead
of passing an iterable of :class:`~oneflow.autograd.Variable` s, pass in an iterable of
:class:`dict` s. Each of them will define a separate parameter group, and should contain
a ``params`` key, containing a list of parameters belonging to it. Other keys
should match the keyword arguments accepted by the optimizers, and will be used
as optimization options for this group.

.. note::

    You can still pass options as keyword arguments. They will be used as
    defaults, in the groups that didn't override them. This is useful when you
    only want to vary a single option, while keeping all others consistent
    between parameter groups.


For example, this is very useful when one wants to specify per-layer learning rates::

    optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)

This means that ``model.base``'s parameters will use the default learning rate of ``1e-2``,
``model.classifier``'s parameters will use a learning rate of ``1e-3``, and a momentum of
``0.9`` will be used for all parameters.

Taking an optimization step
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All optimizers implement a :func:`~Optimizer.step` method, that updates the
parameters. It can be used in two ways:

``optimizer.step()``
~~~~~~~~~~~~~~~~~~~~

This is a simplified version supported by most optimizers. The function can be
called once the gradients are computed using e.g.
:func:`~oneflow.autograd.Variable.backward`.

Example::

    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

``optimizer.step(closure)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some optimization algorithms such as Conjugate Gradient and LBFGS need to
reevaluate the function multiple times, so you have to pass in a closure that
allows them to recompute your model. The closure should clear the gradients,
compute the loss, and return it.

Example::

    for input, target in dataset:
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)

.. _optimizer-algorithms:

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
    SGD

Adjust Learning Rate
--------------------

:mod:`oneflow.optim.lr_scheduler` provides several methods to adjust the learning
rate based on the number of epochs. :class:`oneflow.optim.lr_scheduler.ReduceLROnPlateau`
allows dynamic learning rate reducing based on some validation measurements.

Learning rate scheduling should be applied after optimizer's update; e.g., you
should write your code this way:

Example::

    model = [Parameter(oneflow.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

Most learning rate schedulers can be called back-to-back (also referred to as
chaining schedulers). The result is that each scheduler is applied one after the
other on the learning rate obtained by the one preceding it.

Example::

    model = [Parameter(oneflow.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.1)
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    for epoch in range(20):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler1.step()
        scheduler2.step()

In many places in the documentation, we will use the following template to refer to schedulers
algorithms.

    >>> scheduler = ...
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

.. warning::
  Prior to Oneflow 1.1.0, the learning rate scheduler was expected to be called before
  the optimizer's update; 1.1.0 changed this behavior in a BC-breaking way.  If you use
  the learning rate scheduler (calling ``scheduler.step()``) before the optimizer's update
  (calling ``optimizer.step()``), this will skip the first value of the learning rate schedule.
  If you are unable to reproduce results after upgrading to Oneflow 1.1.0, please check
  if you are calling ``scheduler.step()`` at the wrong time.

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
    lr_scheduler.ConstantLR
    lr_scheduler.LinearLR
    lr_scheduler.ChainedScheduler
    lr_scheduler.SequentialLR
    lr_scheduler.CosineAnnealingWarmRestarts