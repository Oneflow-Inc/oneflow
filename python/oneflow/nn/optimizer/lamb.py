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
from typing import Callable, Dict, Iterator, List, Union, Tuple

import math
import oneflow as flow
from oneflow.optim.optimizer import Optimizer
from oneflow.nn.parameter import Parameter


class LAMB(Optimizer):
    """Implements LAMB algorithm.

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    The equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{u} = \\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}
        
        & \\hat{r} = learning\\_rate * \\frac{||param_{old}||_2}{||\\hat{u}||_2}

        & param_{new} = param_{old} - \\hat{r} * \\hat{u}

    Args:
        parameters (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam_w_mode (bool, optional): apply L2 regularization or weight decay True for
            decoupled weight decay (also known as AdamW) (default: True)
        do_bias_correction (bool, optional): whether to do bias correction (default: True)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm. 
            NOT SUPPORTED now! (default: False)
        contiguous_params (bool, optional): whether to use contiguous ParamGroup 
            which puts all parameters of the same type, device and group into the
            same tensor and update them together. (default: False)
        
    .. _Large Batch Optimization for Deep Learning\\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962

    For example:

    Example 1:

    .. code-block:: python

        # Assume net is a custom model.
        lamb = flow.optim.LAMB(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on.
            # ...
            loss.backward()
            lamb.step()
            lamb.zero_grad()

    Example 2:

    .. code-block:: python

        # Assume net is a custom model.
        lamb = flow.optim.LAMB(
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
            lamb.clip_grad()
            lamb.step()
            lamb.zero_grad()

    If you want to use clip_grad, you can refer this example.

    For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`.
    """

    def __init__(
        self,
        params: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        adam_w_mode: bool = True,
        do_bias_correction: bool = True,
        amsgrad: bool = False,
        contiguous_params: bool = False,
    ):
        if amsgrad:
            # TODO: supported amsgrad in Lamb
            raise RuntimeError("LAMB does not support AMSGrad variant.")
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
        options["adam_w_mode"] = adam_w_mode
        options["bias_correction1"] = 1.0
        options["bias_correction2"] = 1.0
        options["do_bias_correction"] = do_bias_correction
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

        self._op = (
            flow.stateful_op("lamb_update")
            .Input("model")
            .Input("model_diff")
            .Input("m")
            .Input("v")
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
                if param_group["do_bias_correction"]:
                    param_group["bias_correction1"] = 1.0 - math.pow(
                        param_group["betas"][0], self.state["step"] + 1
                    )
                    param_group["bias_correction2"] = 1.0 - math.pow(
                        param_group["betas"][1], self.state["step"] + 1
                    )

                kwargs = {
                    "learning_rate": param_group["lr"],
                    "bias_correction1": param_group["bias_correction1"],
                    "bias_correction2": param_group["bias_correction2"],
                    "beta1": param_group["betas"][0],
                    "beta2": param_group["betas"][1],
                    "epsilon": param_group["eps"],
                    "do_bias_correction": param_group["do_bias_correction"],
                }
                if param_group["adam_w_mode"]:
                    kwargs["weight_decay"] = param_group["weight_decay"]
                    kwargs["l2"] = 0.0
                else:
                    kwargs["l2"] = param_group["weight_decay"]
                    kwargs["weight_decay"] = 0.0

                if param_group["contiguous_params"]:
                    param_list = param_group.contiguous_parameters
                else:
                    param_list = param_group.parameters

                for param in param_list:
                    if param.grad is None:
                        continue
                    if "exp_avg" not in self.state[param]:
                        self.state[param]["exp_avg"] = flow.zeros_like(param)
                    if "exp_avg_sq" not in self.state[param]:
                        self.state[param]["exp_avg_sq"] = flow.zeros_like(param)
                    m_tensor = self.state[param]["exp_avg"]
                    v_tensor = self.state[param]["exp_avg_sq"]

                    flow._C.dispatch_lamb_update(
                        self._op, (param, param.grad, m_tensor, v_tensor), **kwargs
                    )

            self.state["step"] += 1

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
            adam_w_mode = param_group["adam_w_mode"]
            weight_decay = param_group["weight_decay"]
            beta1 = param_group["betas"][0]
            beta2 = param_group["betas"][1]
            do_bias_correction = param_group["do_bias_correction"]
            epsilon = param_group["eps"]

            optimizer_conf.base_learning_rate = lr
            self._generate_lr_scale_for_optim_conf(param_group, optimizer_conf)

            optimizer_conf.lamb_conf.beta1 = beta1
            optimizer_conf.lamb_conf.beta2 = beta2
            optimizer_conf.lamb_conf.epsilon = epsilon
            optimizer_conf.lamb_conf.do_bias_correction = do_bias_correction

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            if adam_w_mode:
                optimizer_conf.weight_decay_conf.weight_decay_rate = weight_decay
            else:
                optimizer_conf.weight_decay_conf.weight_decay_rate = 0.0

            for param in param_group.parameters:
                if not adam_w_mode:
                    # Set l2 penalty as weight decay if **NOT** using adam_w_mode
                    vars_conf[param].l2 = weight_decay
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs
