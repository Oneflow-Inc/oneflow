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
import math
from typing import Callable, Dict, Iterator, List, Tuple, Union

import oneflow as flow
from oneflow.nn.optimizer.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter

from oneflow.nn.optimizer.adamw import AdamW as NNAdamW


class AdamW(NNAdamW):
    """Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    The optimizer of the Adam-weight-decay algorithm.

    (More details please refer to `Adam-weight-decay <https://www.fast.ai/2018/07/02/adam-weight-decay/>`_).

    So we use Adam-weight-decay algorithm to solve this problem.

    the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*(\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}+\\lambda*param_{old})

        & param_{new} = param_{old} - \\hat{g}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (In the equation is λ, default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm. (default: False) 
        do_bias_correction (bool, optional): Whether do bias correction (default: True)

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    For example: 

    Example 1: 

    .. code-block:: python 

        # Assume net is a custom model. 
        adamw = flow.optim.AdamW(net.parameters(), lr=1e-3)

        for epoch in range(epochs):
            # Read data, Compute the loss and so on. 
            # ...
            loss.backward()
            adamw.step()
            adamw.zero_grad()

    Example 2: 

    .. code-block:: python 

        # Assume net is a custom model. 
        adamw = flow.optim.AdamW(
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
            adamw.clip_grad()
            adamw.step()
            adamw.zero_grad()

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
        amsgrad: bool = False,
        do_bias_correction: bool = True,
    ):
        super().__init__(
            params, lr, betas, eps, weight_decay, amsgrad, do_bias_correction
        )

        self._state["step"] = flow.tensor(0)
        self._op_with_amsgrad = (
            flow.stateful_op("adam_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Input("bias_correction1")
            .Input("bias_correction2")
            .Input("m")
            .Input("v")
            .Input("max_v")
            .Build()
        )
        self._op_without_amsgrad = (
            flow.stateful_op("adam_update")
            .Input("model")
            .Input("model_diff")
            .Input("skip_if")
            .Input("bias_correction1")
            .Input("bias_correction2")
            .Input("m")
            .Input("v")
            .Build()
        )

    def step(self, skip_condition=False, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert isinstance(
            skip_condition, (flow.Tensor, bool)
        ), "skip_condition must be  a scalar bool Tensor ora boolean value!"
        if isinstance(skip_condition, flow.Tensor):
            assert (
                skip_condition.dtype is flow.bool and skip_condition.dim() == 0
            ), "skip_condition must be  a scalar bool Tensor ora boolean value!"
        with flow.no_grad():
            if isinstance(skip_condition, bool):
                skip_condition = flow.tensor(skip_condition, dtype=flow.int64)
            else:
                skip_condition = skip_condition.to(flow.int64)
            loss = None
            if closure is not None:
                loss = closure()
            for param_group in self.param_groups:
                if param_group["do_bias_correction"]:
                    param_group["bias_correction1"] = 1.0 - flow.pow(
                        param_group["betas"][0], self._state["step"] + 1
                    )
                    param_group["bias_correction2"] = 1.0 - flow.pow(
                        param_group["betas"][1], self._state["step"] + 1
                    )

                kwargs = {
                    "learning_rate": param_group["lr"],
                    "weight_decay": param_group["weight_decay"],
                    "beta1": param_group["betas"][0],
                    "beta2": param_group["betas"][1],
                    "epsilon": param_group["eps"],
                    "do_bias_correction": param_group["do_bias_correction"],
                    "amsgrad": param_group["amsgrad"],
                }

                for param in param_group.parameters:
                    if param.grad is None:
                        continue

                    if "exp_avg" not in self._state[param]:
                        self._state[param]["exp_avg"] = flow.zeros_like(param)
                    if "exp_avg_sq" not in self._state[param]:
                        self._state[param]["exp_avg_sq"] = flow.zeros_like(param)
                    if param_group["amsgrad"]:
                        if "max_exp_avg_sq" not in self._state[param]:
                            self._state[param]["max_exp_avg_sq"] = flow.zeros_like(
                                param
                            )
                    m_tensor = self._state[param]["exp_avg"]
                    v_tensor = self._state[param]["exp_avg_sq"]

                    if param_group["amsgrad"]:
                        max_v_tensor = self._state[param]["max_exp_avg_sq"]
                        flow._C.dispatch_adam_update(
                            self._op_with_amsgrad,
                            (
                                param,
                                param.grad,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                param_group["bias_correction1"].to(param.device)
                                if param.is_local
                                else param_group["bias_correction1"].to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                param_group["bias_correction2"].to(param.device)
                                if param.is_local
                                else param_group["bias_correction2"].to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                m_tensor,
                                v_tensor,
                                max_v_tensor,
                            ),
                            **kwargs,
                        )
                    else:
                        flow._C.dispatch_adam_update(
                            self._op_without_amsgrad,
                            (
                                param,
                                param.grad,
                                skip_condition.to(param.device)
                                if param.is_local
                                else skip_condition.to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                param_group["bias_correction1"].to(param.device)
                                if param.is_local
                                else param_group["bias_correction1"].to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                param_group["bias_correction2"].to(param.device)
                                if param.is_local
                                else param_group["bias_correction2"].to_global(
                                    param.placement,
                                    [flow.sbp.broadcast for _ in range(len(param.sbp))],
                                ),
                                m_tensor,
                                v_tensor,
                            ),
                            **kwargs,
                        )

            self._state["step"] += skip_condition
            return loss
