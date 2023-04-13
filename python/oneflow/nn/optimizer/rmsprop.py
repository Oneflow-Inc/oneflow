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

import oneflow as flow
from oneflow.optim.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter


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
        contiguous_params (bool, optional): whether to use contiguous ParamGroup 
            which puts all parameters of the same type, device and group into the
            same tensor and update them together. (default: False)

    For example: 

    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        rmsprop = flow.optim.RMSprop(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            rmsprop.step()
            rmsprop.zero_grad()

    Example 2: 

    .. code-block:: python 

        # Assume net is a custom model. 
        rmsprop = flow.optim.RMSprop(
            [
                {
                    "params": net.parameters(),
                    "lr": learning_rate,
                    "clip_grad_max_norm": 0.5,
                    "clip_grad_norm_type": 2.0,
                }
            ],
        )

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            rmsprop.clip_grad()
            rmsprop.step()
            rmsprop.zero_grad()

    If you want to use clip_grad, you can refer this example. 

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`. 

    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0.0,
        centered: bool = False,
        contiguous_params: bool = False,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert alpha >= 0.0, f"Invalid alpha value: {alpha}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert momentum == 0.0, "Not support momentum greater than zeros now!"
        options = dict()
        options["lr"] = lr
        options["alpha"] = alpha
        options["eps"] = eps
        options["weight_decay"] = weight_decay
        options["centered"] = centered
        options["contiguous_params"] = contiguous_params
        super().__init__(params, options)

        for param_group in self.param_groups:
            if param_group["contiguous_params"]:
                param_list = param_group.contiguous_parameters
            else:
                param_list = param_group.parameters

            for param in param_list:
                assert param.is_leaf, "parameters must be leaf tensor"
                self.state[param] = dict()

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
                with flow.enable_grad():
                    loss = closure()

            for param_group in self.param_groups:
                kwargs = {
                    "learning_rate": param_group["lr"],
                    "epsilon": param_group["eps"],
                    "decay_rate": param_group["alpha"],
                    "l2": param_group["weight_decay"],
                }

                if param_group["contiguous_params"]:
                    param_list = param_group.contiguous_parameters
                else:
                    param_list = param_group.parameters

                for param in param_list:
                    if param.grad is None:
                        continue

                    if "square_avg" not in self.state[param]:
                        self.state[param]["square_avg"] = flow.zeros_like(param)
                    ms_tensor = self.state[param]["square_avg"]

                    if param_group["centered"]:
                        if "grad_avg" not in self.state[param]:
                            self.state[param]["grad_avg"] = flow.zeros_like(param)
                        mg_tensor = self.state[param]["grad_avg"]
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
            self.state["step"] = self.state["step"] + 1
            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            assert (
                param_group["contiguous_params"] != True
            ), "contiguous_params cannot be used in graph"

            optimizer_conf = train_conf.optimizer_conf.add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            decay_rate = param_group["alpha"]
            centered = param_group["centered"]
            weight_decay = param_group["weight_decay"]

            epslion = param_group["eps"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.rmsprop_conf.decay_rate = decay_rate
            optimizer_conf.rmsprop_conf.centered = centered
            optimizer_conf.rmsprop_conf.epsilon = epslion

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            # Set l2 penalty as weight decay
            for param in param_group.parameters:
                vars_conf[param].l2 = weight_decay
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs
