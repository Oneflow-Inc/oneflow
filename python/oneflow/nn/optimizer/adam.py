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


class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of gradient.

    the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}

        & param_{new} = param_{old} - \\hat{g}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale (float, optional): the scale factor of loss (default: 1.0)
        do_bias_correction (bool, optional): Whether do bias correction (default: False)

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """

    def __init__(
        self,
        parameters: Union[Iterator[Parameter], List[Dict]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
        scale: float = 1.0,
        do_bias_correction: bool = False,
    ):
        super().__init__()
        assert lr >= 0.0, f"Invalid learning rate: {lr}"
        assert eps >= 0.0, f"Invalid epsilon value: {eps}"
        assert (
            betas[0] >= 0.0 and betas[0] < 1.0
        ), f"Invalid beta parameter at index 0: {betas[0]}"
        assert (
            betas[1] >= 0.0 and betas[1] < 1.0
        ), f"Invalid beta parameter at index 1: {betas[1]}"
        assert weight_decay >= 0.0, f"Invalid weight_decay value: {weight_decay}"
        assert scale > 0.0, f"Invalid scale factor: {scale}"
        assert amsgrad is False, "Not support AMSGrad now!"
        self.do_bias_correction = do_bias_correction
        self._default_options["lr"] = lr
        self._default_options["eps"] = eps
        self._default_options["betas"] = betas
        self._default_options["weight_decay"] = weight_decay
        self._default_options["amsgrad"] = amsgrad
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
                self._state[param]["exp_avg"] = flow.zeros_like(param)
                self._state[param]["exp_avg_sq"] = flow.zeros_like(param)
        self._op = (
            flow.builtin_op("adam_update")
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
                    "scale": param_group["scale"],
                    "l2": param_group["weight_decay"],
                    "beta1": param_group["betas"][0],
                    "beta2": param_group["betas"][1],
                    "epsilon": param_group["eps"],
                }
                if self.do_bias_correction:
                    kwargs["learning_rate_val"] = (
                        param_group["lr"]
                        * math.sqrt(1 - kwargs["beta2"] ** (self._state["step"] + 1))
                        / (1 - kwargs["beta1"] ** (self._state["step"] + 1))
                    )
                for param in param_group.parameters:
                    if param.grad is None:
                        continue
                    m_tensor = self._state[param]["exp_avg"]
                    v_tensor = self._state[param]["exp_avg_sq"]
                    self._op(param, param.grad, m_tensor, v_tensor, **kwargs)
            self._state["step"] = self._state["step"] + 1
            return loss

    def generate_conf_for_graph(self, train_conf, vars_conf):
        for param_group in self.param_groups:
            optimizer_conf = train_conf.mutable_optimizer_conf().Add()

            lr = param_group["lr"]
            scale = param_group["scale"]
            l2 = param_group["weight_decay"]
            beta1 = param_group["betas"][0]
            beta2 = param_group["betas"][1]

            epsilon = param_group["eps"]
            # TODO(): optimizer_conf need to have loss_scale_factor field to support multi scale factor
            base_scale = train_conf.loss_scale_factor()
            assert math.isclose(base_scale, 1, rel_tol=1e-4) or math.isclose(
                scale, base_scale, rel_tol=1e-4
            ), "nn.Graph only support one scale factor at the moment, base_scale {} vs scale {}".format(
                base_scale, scale
            )

            train_conf.set_loss_scale_factor(scale)
            optimizer_conf.set_base_learning_rate(lr)

            optimizer_conf.mutable_adam_conf().set_beta1(beta1)
            optimizer_conf.mutable_adam_conf().set_beta2(beta2)
            optimizer_conf.mutable_adam_conf().set_epsilon(epsilon)
            optimizer_conf.mutable_adam_conf().set_do_bias_correction(
                self.do_bias_correction
            )  # TODO(zzk): Check this option

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.add_variable_op_names(vars_conf[param].name)
