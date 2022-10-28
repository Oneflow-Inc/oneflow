oneflow.optim
===================================

.. The documentation is referenced from: 
   https://pytorch.org/docs/1.10/optim.html

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

.. note::
    If you need to move a model to GPU via ``.cuda()``, please do so before 
    constructing optimizers for it. Parameters of a model after ``.cuda()`` 
    will be different objects with those before the call.

    In general, you should make sure that optimized parameters live in 
    consistent locations when optimizers are constructed and used. 
    
Example::

    import oneflow
    import oneflow.nn as nn
    import oneflow.optim as optim

    model = nn.Linear(16, 3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

Per-parameter options
^^^^^^^^^^^^^^^^^^^^^

:class:`Optimizer` also support specifying per-parameter options. To do this, instead
of passing an iterable of :class:`~oneflow.autograd.Variable`, pass in an iterable of
:class:`dict`. Each of them will define a separate parameter group, and should contain
a ``params`` key, containing a list of parameters belonging to it. Other keys
should match the keyword arguments accepted by the optimizers, and will be used
as optimization options for this group.

.. note::

    You can still pass options as keyword arguments. They will be used as
    defaults, in the groups that didn't override them. This is useful when you
    only want to vary a single option, while keeping all others consistent
    between parameter groups.


For example, this is very useful when one wants to specify per-layer learning rates::

    import oneflow.nn as nn
    import oneflow.optim as optim


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.base = nn.Linear(64, 32)
            self.classifier = nn.Linear(32, 10)

        def forward(self, x):
            out = self.base(x)
            out = self.classifier(out)
            return out


    model = Model()
    optim.SGD(
        [
            {"params": model.base.parameters()},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        lr=1e-2,
        momentum=0.9,
    )


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

    import oneflow
    import oneflow.nn as nn
    import oneflow.nn.functional as F
    import oneflow.optim as optim
    from oneflow.utils.data import Dataset, DataLoader


    class CustomDataset(Dataset):
        def __init__(self, num):
            self.inputs = oneflow.randn(num, 1)
            self.targets = oneflow.sin(self.inputs)

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, index):
            return self.inputs[index], self.targets[index]


    class Model(nn.Module):
        def __init__(self, input_size):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(input_size, 64)
            self.linear2 = nn.Linear(64, input_size)

        def forward(self, x):
            out = self.linear1(x)
            return self.linear2(F.relu(out))


    dataset = CustomDataset(10000)
    dataloader = DataLoader(dataset, batch_size=10)
    model = Model(1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(100):
        for input, target in dataloader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

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

    import oneflow
    import oneflow.nn as nn
    import oneflow.nn.functional as F
    import oneflow.optim as optim
    from oneflow.utils.data import Dataset, DataLoader


    class CustomDataset(Dataset):
        def __init__(self, num):
            self.inputs = oneflow.randn(num, 1)
            self.targets = oneflow.sin(self.inputs)

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, index):
            return self.inputs[index], self.targets[index]


    class Model(nn.Module):
        def __init__(self, input_size):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(input_size, 64)
            self.linear2 = nn.Linear(64, input_size)

        def forward(self, x):
            out = self.linear1(x)
            return self.linear2(F.relu(out))


    dataset = CustomDataset(10000)
    dataloader = DataLoader(dataset, batch_size=10)
    model = Model(1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(20):
        for input, target in dataloader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

Most learning rate schedulers can be chained (also referred to as
chaining schedulers).

Example::

    import oneflow
    import oneflow.nn as nn
    import oneflow.nn.functional as F
    import oneflow.optim as optim
    from oneflow.utils.data import Dataset, DataLoader


    class CustomDataset(Dataset):
        def __init__(self, num):
            self.inputs = oneflow.randn(num, 1)
            self.targets = oneflow.sin(self.inputs)

        def __len__(self):
            return self.inputs.shape[0]

        def __getitem__(self, index):
            return self.inputs[index], self.targets[index]


    class Model(nn.Module):
        def __init__(self, input_size):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(input_size, 64)
            self.linear2 = nn.Linear(64, input_size)

        def forward(self, x):
            out = self.linear1(x)
            return self.linear2(F.relu(out))


    dataset = CustomDataset(10000)
    dataloader = DataLoader(dataset, batch_size=10)
    model = Model(1)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

    for epoch in range(20):
        for input, target in dataloader:
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
  If you use the learning rate scheduler (calling ``scheduler.step()``) before the optimizer's update
  (calling ``optimizer.step()``), this will skip the first value of the learning rate schedule. Please 
  check if you are calling ``scheduler.step()`` at the wrong time.

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
