import math
from .lr_scheduler import LrScheduler

class CosineAnnealingLR(LrScheduler):
    """This operator creates a Cosine decayed learning rate scheduler.

    Before the steps are specified by user, the learning rate will be updated as:

    .. math::

        & cos\\_decay = 0.5*(1+cos(\\pi*\\frac{current\\_step}{steps}))

        & decay\\_factor = (1-\\alpha)*cos\\_decay+\\alpha

        & learning\\_rate = base\\_learning\\_rate*decay\\_factor

    After the steps specified by user, the learning rate will be :

    .. math::

        learning\\_rate = {base\\_learning\\_rate}*{\\alpha}

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer(Optimizer): Wrapped optimizer.
        steps (int): The decay steps in the scheduler.
        alpha (float, optional): The learning rate scale factor (:math:`\\alpha`). (default: 0.0)
        last_step (int, optional): The index of last step. (default: -1)
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. (default: ``False``)

    For example:

    .. code-block:: python

        import oneflow as flow

        ...
        cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps=100, alpha=0.0)
        for epoch in range(num_epoch):
            train(...)
            cosine_annealing_lr.step()

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, steps: int, alpha: float=0.0, last_step=-1, verbose=False):
        assert steps > 0, f'steps must greater than zero, but got {steps}'
        self.steps = steps
        self.alpha = alpha
        super().__init__(optimizer, last_step, verbose)

    def get_lr(self):
        if self.last_step < self.steps:
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_step / self.steps))
            decay_factor = (1 - self.alpha) * cos_decay + self.alpha
            return [base_lr * decay_factor for base_lr in self.base_lrs]
        else:
            return [base_lr * self.alpha for base_lr in self.base_lrs]