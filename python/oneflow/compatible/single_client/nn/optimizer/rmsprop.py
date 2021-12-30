"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import collections
from typing import Callable, Dict, Iterator, List, Union

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.optimizer.optimizer import (
    Optimizer,
    ParamGroup,
)
from oneflow.compatible.single_client.nn.parameter import Parameter


class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    oot Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        W = w - \\frac{\\eta} {\\\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.
    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    if centered is True:

    .. math::

        r(w, t) = \\alpha r(w, t-1) + (1 - \\alpha)(\\nabla Q_{i}(w))^2

        g(w, t) = \\alpha g(w, t-1) + (1 - \\alpha)\\nabla Q_{i}(w)

        v(w, t) = \\beta v(w, t-1) + \\frac{\\eta} {\\\\sqrt{r(w,t) - (g(w, t))^2 +
            \\epsilon}} \\nabla Q_{i}(w)

        w = w - v(w, t)

    where, :math:`\\alpha` is a hyperparameter and typical values are 0.99, 0.95
    and so on. :math:`\\beta` is the momentum term. :math:`\\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0, oneflow not support momenmtum > 0 now!)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0.0,
        centered: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert alpha >= 0.0, f"Invalid alpha value: {alpha}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert scale > 0.0, f"Invalid scale factor: {scale}"
        assert momentum == 0.0, "Not support momentum greater than zeros now!"
        self._default_options["lr"] = lr
        self._default_options["alpha"] = alpha
        self._default_options["eps"] = eps
        self._default_options["weight_decay"] = weight_decay
        self._default_options["centered"] = centered
        self._default_options["scale"] = scale
        if isinstance(parameters, collections.abc.Iterator):
            self.param_groups.append(ParamGroup(parameters, self._default_options))
        else:
            for param in parameters:
                self.param_groups.append(ParamGroup(param, self._default_options))
        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
                self._state[param]["square_avg"] = flow.experimental.zeros_like(param)
                if param_group["centered"]:
                    self._state[param]["grad_avg"] = flow.experimental.zeros_like(param)
        self._centered_rmsprop = (
            flow.stateful_op("rmsprop_update")
            .Input("model")
            .Input("model_diff")
            .Input("mean_square")
            .Input("mean_gradient")
            .Build()
        )
        self._rmsprop = (
            flow.stateful_op("rmsprop_update")
            .Input("model")
            .Input("model_diff")
            .Input("mean_square")
            .Build()
        )

    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with flow.no_grad():
            loss = None
            if closure is not None:
                loss = closure()
            for param_group in self.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "scale": param_group["scale"],
                    "epsilon": param_group["eps"],
                    "decay_rate": param_group["alpha"],
                    "weight_decay": param_group["weight_decay"],
                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    ms_tensor = self._state[param]["square_avg"]
                    if param_group["centered"]:
                        mg_tensor = self._state[param]["grad_avg"]
                        flow._C.dispatch_rmsprop_update(
                            self._centered_rmsprop,
                            (param, param.grad, ms_tensor, mg_tensor),
                            centered=True,
                            **kwargs,
                        )
                    else:
                        flow._C.dispatch_rmsprop_update(
                            self._rmsprop, (param, param.grad, ms_tensor), **kwargs
                        )
            self._state["step"] = self._state["step"] + 1
            return loss
