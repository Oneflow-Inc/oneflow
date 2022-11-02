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
from oneflow.nn.parameter import Parameter

from oneflow.nn.optimizer.lamb import LAMB as NNLAMB


class LAMB(NNLAMB):
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
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            adam_w_mode,
            do_bias_correction,
            amsgrad,
        )

        self._state["step"] = flow.tensor(0)
        self._op = (
            flow.stateful_op("lamb_update")
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
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    if "exp_avg" not in self._state[param]:
                        self._state[param]["exp_avg"] = flow.zeros_like(param)
                    if "exp_avg_sq" not in self._state[param]:
                        self._state[param]["exp_avg_sq"] = flow.zeros_like(param)
                    m_tensor = self._state[param]["exp_avg"]
                    v_tensor = self._state[param]["exp_avg_sq"]

                    flow._C.dispatch_lamb_update(
                        self._op,
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
                        **kwargs
                    )

            self._state["step"] += skip_condition

            return loss
