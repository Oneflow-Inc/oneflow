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
from typing import Callable, Dict, Iterator, List, Union, Tuple

import oneflow as flow
from oneflow.nn.optimizer.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter


class LAMB(Optimizer):
    """Implements LAMB algorithm.

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
    https://arxiv.org/abs/1904.00962

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
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
        do_bias_correction: bool = True,
    ):
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert (
            betas[0] >= 0.0 and betas[0] < 1.0
        ), f"Invalid beta parameter at index 0: {betas[0]}"
        assert (
            betas[1] >= 0.0 and betas[1] < 1.0
        ), f"Invalid beta parameter at index 1: {betas[1]}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"

        options = dict()
        options["lr"] = lr
        options["eps"] = eps
        options["betas"] = betas
        options["weight_decay"] = weight_decay
        options["amsgrad"] = amsgrad
        options["bias_correction1"] = 1.0
        options["bias_correction2"] = 1.0
        options["do_bias_correction"] = do_bias_correction

        super().__init__(parameters, options)

        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()
        
        self._lamb_op = (
            flow.builtin_op("lamb_update")
            .Input("model")
            .Input("model_diff")
            .Input("m")
            .Input("v")
            .Attr("l1", 0.0)
            .Attr("weight_decay", 0.0)
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
                    "learning_rate_val": param_group["lr"],
                    "bias_correction1_val": param_group["bias_correction1"],
                    "bias_correction2_val": param_group["bias_correction2"],
                    "decay_rate": param_group["alpha"],
                    "l2": param_group["weight_decay"],
                    "beta1": param_group["betas"][0],
                    "beta2": param_group["betas"][1],
                    "epsilon": param_group["eps"],
                    "do_bias_correction": param_group["do_bias_correction"],

                }
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "exp_avg" not in self._state[param]:
                        self._state[param]["exp_avg"] = flow.zeros_like(param)
                    if "exp_avg_sq" not in self._state[param]:
                        self._state[param]["exp_avg_sq"] = flow.zeros_like(param)
                    m_tensor = self._state[param]["exp_avg"]
                    v_tensor = self._state[param]["exp_avg_sq"]
                    
                    self._lamb_op(
                        param, param.grad, m_tensor, v_tensor
                    )
            self._state["step"] = self._state["step"] + 1
            return loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        new_opt_confs = []
        for param_group in self.param_groups:
            optimizer_conf = train_conf.mutable_optimizer_conf().Add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            l2 = param_group["weight_decay"]
            beta1 = param_group["betas"][0]
            beta2 = param_group["betas"][1]
            
            epsilon = param_group["eps"]

            optimizer_conf.set_base_learning_rate(lr)

            optimizer_conf.mutable_lamb_conf().set_beta1(beta1)
            optimizer_conf.mutable_lamb_conf().set_beta2(beta2)
            optimizer_conf.mutable_lamb_conf().set_epsilon(epsilon)

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            # Set l2 penalty as weight decay
            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.add_variable_op_names(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs
