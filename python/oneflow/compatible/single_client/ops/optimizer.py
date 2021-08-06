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
import collections.abc
import traceback
from typing import Callable, List, Optional, Sequence, Text, Union

import oneflow._oneflow_internal
from oneflow import oneflow_deprecate
from oneflow._oneflow_internal.oneflow.core.job import job_conf as job_conf_cfg
from oneflow._oneflow_internal.oneflow.core.job import (
    learning_rate_schedule_conf as learning_rate_schedule_conf_cfg,
)
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import runtime_mode as rt_mode
from oneflow.compatible.single_client.framework import session_context as session_ctx


def GetVariablesForCurrentJob() -> List[Text]:
    sess = session_ctx.GetDefaultSession()
    assert (
        rt_mode.CurrentMode() == rt_mode.GLOBAL_MODE
    ), "Optimizer's Variables() or minimize() method should be called inside a Job Function to implicitly get variables from a job."
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    return list(sess.job_name2var_name2var_blob_[job_name].keys())


class ClipGradientConf:
    @property
    def clip_conf(self) -> job_conf_cfg.ClipConf:
        raise NotImplementedError()


class by_global_norm(ClipGradientConf):
    """This operator limits the norm of `Input` with `clip_norm`.

    If the norm of `Input` is less than the `clip_norm`,

    the `Output` will be the same as `Input`.

    If the norm of `Input` is greater than the `clip_norm`, the `Output` will be scaled.

    The equation is:

    .. math::

        Output = \\frac{clip\\_norm*Input}{norm(Input)}

    Args:
        clip_norm (float): The maximum norm value.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set gradient_clip
            gradient_clip = flow.optimizer.grad_clipping.by_global_norm(1.0)
            # Set AdamW optimizer with gradient clip
            flow.optimizer.AdamW(lr_scheduler,
                        do_bias_correction=False, weight_decay=0.00005,
                        grad_clipping=gradient_clip).minimize(loss)

            return loss

    """

    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    @property
    def clip_conf(self):
        clip_conf = job_conf_cfg.ClipConf()
        clip_conf.mutable_clip_by_global_norm().set_clip_norm(self.clip_norm)
        return clip_conf


class WarmupConf:
    @property
    def warmup_conf(self) -> learning_rate_schedule_conf_cfg.WarmupConf:
        raise NotImplementedError()


class constant(WarmupConf):
    """This operator use the constant warmup strategy to adjust the learning rate.

    Before the steps are specified by user, the learning rate is:

    .. math::

        learning\\_rate = base\\_learning\\_rate*multiplier

    After the steps are specified by user, the learning rate is:

    .. math::

        learning\\_rate = base\\_learning\\_rate

    Args:
        steps (int): [description]
        multiplier (float): The scale factor :math:`multiplier`, it should be greater than 0. and less than 1.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Before 10 epochs, the learning rate is 0.001
            # After 10 epochs, the learning rate is 0.01
            warmup_scheduler = flow.optimizer.warmup.constant(10, 0.1)
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01], warmup=warmup_scheduler)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(self, steps, multiplier):
        self.steps = steps
        self.multiplier = multiplier

    @property
    def warmup_conf(self) -> learning_rate_schedule_conf_cfg.WarmupConf:
        warmup_conf = learning_rate_schedule_conf_cfg.WarmupConf()
        warmup_conf.mutable_constant_conf().set_warmup_batches(self.steps)
        warmup_conf.mutable_constant_conf().set_multiplier(self.multiplier)
        return warmup_conf


class linear(WarmupConf):
    """This operator uses the linear warmup strategy to adjust the learning rate.

    When current train step is less than warmup steps, the learning rate will be updated as:

    .. math::

        & current\\_multiplier = start\\_multiplier + (1-start\\_multiplier)*\\frac{train\\_step}{warmup\\_step}

        & current\\_learning\\_rate = learning\\_rate*current\\_multiplier

    Args:
        steps (int): The warmup steps.
        start_multiplier (float): The start multiplier(:math:`start\\_multiplier`). It should be greater than 0. and less than 1.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Before 10 epochs, the learning rate will increase from 0.001 to 0.01 in linear.
            warmup_scheduler = flow.optimizer.warmup.linear(10, 0.1)
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01], warmup=warmup_scheduler)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(self, steps, start_multiplier):
        self.steps = steps
        self.start_multiplier = start_multiplier

    @property
    def warmup_conf(self) -> learning_rate_schedule_conf_cfg.WarmupConf:
        warmup_conf = learning_rate_schedule_conf_cfg.WarmupConf()
        warmup_conf.mutable_linear_conf().set_warmup_batches(self.steps)
        warmup_conf.mutable_linear_conf().set_start_multiplier(self.start_multiplier)
        return warmup_conf


class LrScheduler:
    def __init__(
        self,
        base_lr: Optional[float] = None,
        lr_lbn: Optional[Text] = None,
        warmup: Optional[WarmupConf] = None,
    ):
        self.base_lr = base_lr
        self.lr_lbn = lr_lbn
        self.warmup = warmup

    @property
    def warmup_conf(self) -> learning_rate_schedule_conf_cfg.WarmupConf:
        if self.warmup is None:
            return None
        return self.warmup.warmup_conf

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        raise NotImplementedError()

    def SetLrFieldsInOptimizerConf(self, optimizer_conf) -> None:
        if self.lr_lbn is not None:
            assert self.base_lr is None
            assert self.warmup is None
            assert self.learning_rate_decay_conf is None
            optimizer_conf.set_learning_rate_lbn(self.lr_lbn)
        else:
            assert self.base_lr is not None
            optimizer_conf.set_base_learning_rate(self.base_lr)
            if self.warmup_conf is not None:
                optimizer_conf.mutable_warmup_conf().CopyFrom(self.warmup_conf)
            if self.learning_rate_decay_conf is not None:
                optimizer_conf.mutable_learning_rate_decay().CopyFrom(
                    self.learning_rate_decay_conf
                )


class CosineScheduler(LrScheduler):
    """This operator creates a Cosine decayed learning rate scheduler.

    Before the steps are specified by user, the learning rate will be updated as:

    .. math::

        & cos\\_decay = 0.5*(1+cos(\\pi*\\frac{current\\_batch}{decayed\\_batch}))

        & decay\\_factor = (1-\\alpha)*cos\\_decay+\\alpha

        & learning\\_rate = base\\_learning\\_rate*decay\\_factor

    After the steps specified by user, the learning rate will be :

    .. math::

        learning\\_rate = {base\\_learning\\_rate}*{\\alpha}

    Args:
        base_lr (float): The base learning rate (:math:`base\\_learning\\_rate`)
        steps (int): The decay steps in the scheduler (:math:`decayed\\_batch`)
        alpha (float, optional): The learning rate scale factor (:math:`\\alpha`). Defaults to 0.0.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.CosineScheduler(base_lr=0.01,
                                                          steps=10,
                                                          alpha=0.1)
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        alpha: float = 0.0,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.alpha = alpha

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_cosine_conf().set_decay_batches(self.steps)
        learning_rate_decay_conf.mutable_cosine_conf().set_alpha(self.alpha)
        return learning_rate_decay_conf


class CustomScheduler(LrScheduler):
    def __init__(self, lbn: Text):
        super().__init__(lr_lbn=lbn)

    @property
    def learning_rate_decay_conf(
        self,
    ) -> learning_rate_schedule_conf_cfg.LearningRateDecayConf:
        return None


class PiecewiseConstantScheduler(LrScheduler):
    """This operator creates a piecewise constant learning rate scheduler.

    The change in learning rate can be described as follows:

    .. code-block:: python

        boundaries = [1000, 2000]
        values = [0.1, 0.01, 0.001]

        if current_step < 1000:
            learning_rate = 0.1
        elif 1000 < current_step < 2000:
            learning_rate = 0.01
        else:
            learning_rate = 0.001

    Args:
        boundaries (Sequence[int]): A list of train steps.
        values (Sequence[float]): A list of learning rate values during the different train step boundary.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
                images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler(boundaries=[10, 20],
                                                                     values=[0.1, 0.01, 0.001])
            flow.optimizer.Adam(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(
        self,
        boundaries: Sequence[int],
        values: Sequence[float],
        warmup: Optional[WarmupConf] = None,
    ):
        assert len(boundaries) + 1 == len(values)
        super().__init__(base_lr=values[0], warmup=warmup)
        self.boundaries = boundaries
        self.values = values

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        for boundary in self.boundaries:
            learning_rate_decay_conf.mutable_piecewise_constant_conf().add_boundaries(
                boundary
            )
        for value in self.values:
            learning_rate_decay_conf.mutable_piecewise_constant_conf().add_values(value)
        return learning_rate_decay_conf


class PiecewiseScalingScheduler(LrScheduler):
    """This operator creates a piecewise scaled decayed learning rate scheduler.

    The change in learning rate can be described as follows:

    .. code-block:: python

        boundaries = [1000, 2000]
        scale = [0.1, 0.01]
        base_lr = 0.1

        if current_step < 1000:
            learning_rate = base_lr
        elif 1000 < current_step < 2000:
            learning_rate = 0.1*base_lr
        else:
            learning_rate = 0.01*base_lr

    Args:
        base_lr (float): The base learning rate
        boundaries (Sequence[int]): A list of train steps.
        scale (Union[float, Sequence[float]]): A list of learning rate scaled factors during the different train step boundary.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(base_lr=0.1,
                                                                    boundaries=[5, 10],
                                                                    scale=[0.5, 0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

            return loss

    """

    def __init__(
        self,
        base_lr: float,
        boundaries: Sequence[int],
        scale: Union[float, Sequence[float]],
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.boundaries = boundaries
        if not isinstance(scale, collections.abc.Sequence):
            scale = [scale] * len(boundaries)
        assert len(boundaries) == len(scale)
        self.scales = [1] + list(scale)

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        for boundary in self.boundaries:
            learning_rate_decay_conf.mutable_piecewise_scaling_conf().add_boundaries(
                boundary
            )
        for scale in self.scales:
            learning_rate_decay_conf.mutable_piecewise_scaling_conf().add_scales(scale)
        return learning_rate_decay_conf


class PolynomialScheduler(LrScheduler):
    """This operator creates a polynomial decayed learning rate scheduler.

    The learning rate will be updated as follows:

    If cycle is `True`, the equation is:

    .. math::

        & decay\\_batch = decay\\_batch*ceil(\\frac{current\\_batch}{decay\\_batch})

        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr

    If cycle is `False`, the equation is:

    .. math::

        & decay\\_batch = min(decay\\_batch, current\\_batch)

        & learning\\_rate = (base\\_lr-end\\_lr)*(1-\\frac{current\\_batch}{decay\\_batch})^{pow}+end\\_lr

    Args:
        base_lr (float): The base learning rate
        steps (int): The decayed steps
        end_learning_rate (float, optional): The final learning rate. Defaults to 0.0001.
        power (float, optional): The power of polynomial. Defaults to 1.0.
        cycle (bool, optional): If cycle is true, the scheduler will decay the learning rate every decay steps. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow.compatible.single_client as flow
            import oneflow.compatible.single_client.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.PolynomialScheduler(base_lr=0.001,
                                                                 steps=5,
                                                                 end_learning_rate=0.00001,
                                                                 power=2)
                flow.optimizer.Adam(lr_scheduler).minimize(loss)

                return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_polynomial_conf().set_decay_batches(self.steps)
        learning_rate_decay_conf.mutable_polynomial_conf().set_end_learning_rate(
            self.end_learning_rate
        )
        learning_rate_decay_conf.mutable_polynomial_conf().set_power(self.power)
        learning_rate_decay_conf.mutable_polynomial_conf().set_cycle(self.cycle)
        return learning_rate_decay_conf


from oneflow import oneflow_deprecate


@oneflow_deprecate()
class PolynomialSchduler(PolynomialScheduler):
    def __init__(
        self,
        base_lr: float,
        steps: int,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        print(
            "WARNING:",
            "oneflow.compatible.single_client.optimizer.PolynomialSchduler",
            "will be removed in the future, use {} instead.".format(
                "oneflow.compatible.single_client.optimizer.PolynomialScheduler"
            ),
        )
        print(traceback.format_stack()[-2])
        super().__init__(
            base_lr=base_lr,
            steps=steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=cycle,
            warmup=warmup,
        )


class LinearCosineScheduler(LrScheduler):
    """This operator creates a linear cosine decayed learning rate scheduler.

    The learning rate will be updated as follows:

    .. math::

        & current\\_batch = min(current\\_batch, decay\\_batch)

        & linear\\_decay = \\frac{(decay\\_batch - current\\_batch)}{decay\\_batch}

        & cosine\\_decay = 0.5*(1.0+cos(2*\\pi*num\\_periods*\\frac{current\\_batch}{decay\\_batch}))

        & decay\\_factor = (\\alpha+linear\\_decay)*cosine\\_decay + \\beta

        & learning\\_rate = base\\_learning\\_rate*decay\\_factor

    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        num_periods (float, optional): The number of decay periods. Defaults to 0.5.
        alpha (float, optional): The :math:`\\alpha` in equation. Defaults to 0.0.
        beta (float, optional): The :math:`\\beta` in equation. Defaults to 0.001.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow.compatible.single_client as flow
            import oneflow.compatible.single_client.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.LinearCosineScheduler(base_lr=0.1,
                                                                    steps=10)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        num_periods: float = 0.5,
        alpha: float = 0.0,
        beta: float = 0.001,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_linear_cosine_conf().set_decay_batches(
            self.steps
        )
        learning_rate_decay_conf.mutable_linear_cosine_conf().set_num_periods(
            self.num_periods
        )
        learning_rate_decay_conf.mutable_linear_cosine_conf().set_alpha(self.alpha)
        learning_rate_decay_conf.mutable_linear_cosine_conf().set_beta(self.beta)
        return learning_rate_decay_conf


class ExponentialScheduler(LrScheduler):
    """This operator creates a exponential decayed learning rate scheduler.

    The learning rate will be updated as follows:

    If staircase is set to False, the equation is:

    .. math::

        & pow = \\frac{current\\_batch}{decay\\_batch}

        & learning\\_rate = base\\_learning\\_rate*decay\\_rate^{pow}

    If staircase is set to True, the equation is:

    .. math::

        & pow = floor(\\frac{current\\_batch}{decay\\_batch})

        & learning\\_rate = base\\_learning\\_rate*decay\\_rate^{pow}

    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block::python

            import oneflow.compatible.single_client as flow
            import oneflow.compatible.single_client.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.CosineScheduler(base_lr=0.01,
                                                              steps=10,
                                                              alpha=0.1)
                flow.optimizer.Adam(lr_scheduler).minimize(loss)

                return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase=False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_exponential_conf().set_decay_batches(
            self.steps
        )
        learning_rate_decay_conf.mutable_exponential_conf().set_decay_rate(
            self.decay_rate
        )
        learning_rate_decay_conf.mutable_exponential_conf().set_staircase(
            self.staircase
        )
        return learning_rate_decay_conf


class InverseTimeScheduler(LrScheduler):
    """This operator creates a inverse time decayed learning rate scheduler.

    The learning rate will be updated as follows:

    If staircase is set to False, the equation is:

    .. math::

        & step\\_ratio = \\frac{current\\_batch}{decay\\_batch}

        & learning\\_rate = \\frac{base\\_learning\\_rate}{1+decay\\_rate*step\\_ratio}

    If staircase is set to True, the equation is:

    .. math::

        & step\\_ratio = \\frac{current\\_batch}{decay\\_batch}

        & learning\\_rate = \\frac{base\\_learning\\_rate}{1+floor(decay\\_rate*step\\_ratio)}

    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow.compatible.single_client as flow
            import oneflow.compatible.single_client.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.InverseTimeScheduler(base_lr=0.1,
                                                                   steps=5,
                                                                   decay_rate=0.9)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_inverse_time_conf().set_decay_batches(
            self.steps
        )
        learning_rate_decay_conf.mutable_inverse_time_conf().set_decay_rate(
            self.decay_rate
        )
        learning_rate_decay_conf.mutable_inverse_time_conf().set_staircase(
            self.staircase
        )
        return learning_rate_decay_conf


class NaturalExpScheduler(LrScheduler):
    """This operator creates a natural exponential decayed learning rate scheduler.

    The learning rate will be updated as follows:

    If staircase is set to False, the equation is:

    .. math::

        & step\\_ratio = \\frac{current\\_batch}{decay\\_batch}

        & learning\\_rate = {base\\_learning\\_rate}*e^{-decay\\_rate*step\\_ratio}

    If staircase is set to True, the equation is:

    .. math::

        & step\\_ratio = \\frac{current\\_batch}{decay\\_batch}

        & learning\\_rate = {base\\_learning\\_rate}*e^{-decay\\_rate*floor(step\\_ratio)}

    Args:
        base_lr (float): The base learning rate
        steps (int): The decay steps
        decay_rate (float): The decay rate
        staircase (bool, optional): If staircase is True, the scheduler decay the learning rate at discrete intervals. Defaults to False.
        warmup (Optional[WarmupConf], optional): The warmup strategy. Defaults to None.

    For example:

        .. code-block:: python

            import oneflow.compatible.single_client as flow
            import oneflow.compatible.single_client.typing as tp

            @flow.global_function(type="train")
            def train_job(
                    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
            ) -> tp.Numpy:
                with flow.scope.placement("gpu", "0:0"):
                    logits = lenet(images, train=True)
                    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                        labels, logits, name="softmax_loss"
                    )

                lr_scheduler = flow.optimizer.NaturalExpScheduler(base_lr=0.1,
                                                                  steps=10,
                                                                  decay_rate=0.5)
                flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

                return loss

    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        decay_rate: float,
        staircase: bool = False,
        warmup: Optional[WarmupConf] = None,
    ):
        super().__init__(base_lr=base_lr, warmup=warmup)
        self.steps = steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    @property
    def learning_rate_decay_conf(
        self,
    ) -> Optional[learning_rate_schedule_conf_cfg.LearningRateDecayConf]:
        learning_rate_decay_conf = (
            learning_rate_schedule_conf_cfg.LearningRateDecayConf()
        )
        learning_rate_decay_conf.mutable_natural_exp_conf.set_decay_batches(self.steps)
        learning_rate_decay_conf.mutable_natural_exp_conf.set_decay_rate(
            self.decay_rate
        )
        learning_rate_decay_conf.mutable_natural_exp_conf.set_staircase(self.staircase)
        return learning_rate_decay_conf


class LossScalePolicy:
    def SetLossScaleFieldsInTrainConf(self, train_conf):
        raise NotImplementedError()


class StaticLossScalePolicy(LossScalePolicy):
    def __init__(self, loss_scale_factor: float):
        super().__init__()
        self.loss_scale_factor = loss_scale_factor

    def SetLossScaleFieldsInTrainConf(self, train_conf):
        train_conf.loss_scale_factor = self.loss_scale_factor


class DynamicLossScalePolicy(LossScalePolicy):
    def __init__(
        self, initial_loss_scale=2 ** 30, increment_period=2000, multiplier=2.0
    ):
        super().__init__()
        self.initial_loss_scale = initial_loss_scale
        self.increment_period = increment_period
        self.multiplier = multiplier

    def SetLossScaleFieldsInTrainConf(self, train_conf):
        train_conf.mutable_dynamic_loss_scale_policy().set_initial_loss_scale(
            self.initial_loss_scale
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_increment_period(
            self.increment_period
        )
        train_conf.mutable_dynamic_loss_scale_policy().set_multiplier(self.multiplier)


class Optimizer:
    def __init__(
        self,
        loss_scale_factor: Optional[int] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
    ):
        self.train_step_lbn = train_step_lbn
        if loss_scale_factor is not None:
            assert loss_scale_policy is None
            self.loss_scale_policy = StaticLossScalePolicy(loss_scale_factor)
        else:
            self.loss_scale_policy = loss_scale_policy
        self._variables_list_init = False

    def Variables(self) -> List[Text]:
        if not self._variables_list_init:
            if self.variables is None:
                self.variables = list(GetVariablesForCurrentJob())
            elif callable(self.variables):
                self.variables = list(self.variables())
            else:
                self.variables = list(self.variables)
            self._variables_list_init = True
        return self.variables

    def _AddOptimizerConfInTrainConf(self, train_conf: job_conf_cfg.TrainConf) -> None:
        raise NotImplementedError()

    @property
    def train_conf(self) -> job_conf_cfg.TrainConf:
        train_conf = job_conf_cfg.TrainConf()
        if self.train_step_lbn is not None:
            train_conf.set_train_step_lbn(self.train_step_lbn)
        if self.loss_scale_policy is not None:
            self.loss_scale_policy.SetLossScaleFieldsInTrainConf(train_conf)
        self._AddOptimizerConfInTrainConf(train_conf)
        return train_conf

    def minimize(
        self,
        loss: Union[
            Sequence[oneflow._oneflow_internal.BlobDesc],
            oneflow._oneflow_internal.BlobDesc,
        ],
    ) -> None:
        if not isinstance(loss, collections.abc.Sequence):
            loss = [loss]
        c_api_util.CurJobBuildAndInferCtx_SetTrainConf(self.train_conf)
        for x in loss:
            flow.losses.add_loss(x)


class SGD(Optimizer):
    """The optimizer of the stochastic gradient descent algorithm.

    This algorithm takes a random sample's gradient as an approximate estimate of the overall gradient in small batch gradient descent.

    When the momentum = 0, the equation of parameters updating is:

    .. math::

        param_{new} = param_{old} - learning\\_rate*grad

    With momentum, the equation of parameters updating is:

    .. math::

        & V_{t} = \\beta*V_{t-1} + learning\\_rate*g_t

        & param_{new} = param_{old} - V_{t}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        momentum (float, optional): Momentum factor (:math:`\\beta`). Defaults to 0.9.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Set Learning rate as 0.1
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            # Set Momentum=0.9 SGD optimizer
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(loss)

            return loss
    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        momentum: float = 0.9,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.momentum = momentum
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        if self.momentum == 0:
            optimizer_conf.mutable_naive_conf()
        else:
            optimizer_conf.mutable_momentum_conf().set_beta(self.momentum)
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class SGDW(Optimizer):
    """The optimizer of the stochastic-gradient-descent-weight-decay algorithm.

    (More details please refer to `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_).

    When the momentum = 0, the equation of parameters updating is:

    .. math::

        param_{new} = param_{old} - learning\\_rate*(grad + \\lambda*param_{old}))

    With momentum, the equation of parameters updating is:

    .. math::

        & V_{t} = \\beta*V_{t-1} - learning\\_rate*g_t

        & param_{new} = param_{old} + V_{t} - learning\\_rate * \\lambda*param_{old}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        momentum (float, optional): Momentum factor (:math:`\\beta`). Defaults to 0.9.
        weight_decay (Optional[float], optional): The weight decay factor (In the equation is :math:`\\lambda`). Defaults to None.
        weight_decay_includes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that use weight decay. Defaults to None.
        weight_decay_excludes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that do not use weight decay. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    Note:

        Only one of `weight_decay_includes` and `weight_decay_excludes` can be set. If both are None,
        all the model parameters will use weight decay.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Set Learning rate as 0.1
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            # Set Momentum=0.9 SGDW optimizer, weight_decay factor is 0.00005
            flow.optimizer.SGDW(lr_scheduler, momentum=0.9, weight_decay=0.00005).minimize(loss)

            return loss
    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[float] = None,
        momentum: float = 0.9,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.momentum = momentum
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        if self.momentum == 0:
            optimizer_conf.mutable_naive_conf()
        else:
            optimizer_conf.mutable_momentum_conf().set_beta(self.momentum)
        if self.weight_decay is not None:
            optimizer_conf.mutable_weight_decay_conf().set_weight_decay_rate(
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                for weight_decay_include in self.weight_decay_includes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_includes().add_pattern(
                        weight_decay_include
                    )
            elif self.weight_decay_excludes is not None:
                for weight_decay_exclude in self.weight_decay_excludes:
                    optimizer_conf.weight_decay_conf().mutable_excludes().add_pattern(
                        weight_decay_exclude
                    )
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class Adam(Optimizer):
    """The optimizer of the Adam algorithm.

    This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates

    and the 2nd-moment estimates of gradient.

    With bias correction, the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{V_t} = \\frac{V_t}{1-\\beta_1^t}

        & \\hat{S_t} = \\frac{S_t}{1-\\beta_2^t}

        & \\hat{g} = learning\\_rate*\\frac{\\hat{V_t}}{\\sqrt{\\hat{S_t}}+\\epsilon}

        & param_{new} = param_{old} - \\hat{g}

    Without bias correction, the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}

        & param_{new} = param_{old} - \\hat{g}

    More details please refer to `Adam <https://arxiv.org/abs/1412.6980>`_

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-8.
        do_bias_correction (bool, optional): Whether to do the bias correction. Defaults to False.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set Adam optimizer
            flow.optimizer.Adam(lr_scheduler, do_bias_correction=False).minimize(loss)

            return loss
    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_adam_conf().set_beta1(self.beta1)
        optimizer_conf.mutable_adam_conf().set_beta2(self.beta2)
        optimizer_conf.mutable_adam_conf().set_epsilon(self.epsilon)
        optimizer_conf.mutable_adam_conf().set_do_bias_correction(
            self.do_bias_correction
        )
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class AdamW(Optimizer):
    """The optimizer of the Adam-weight-decay algorithm.

    If we use L2 regularization,

    it will be invalid due to the adaptive learning rate in Adam optimizer

    (More details please refer to `Adam-weight-decay <https://www.fast.ai/2018/07/02/adam-weight-decay/>`_).

    So we use Adam-weight-decay algorithm to solve this problem.

    With bias correction, the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{V_t} = \\frac{V_t}{1-\\beta_1^t}

        & \\hat{S_t} = \\frac{S_t}{1-\\beta_2^t}

        & \\hat{g} = learning\\_rate*(\\frac{\\hat{V_t}}{\\sqrt{\\hat{S_t}}+\\epsilon}+\\lambda*param_{old})

        & param_{new} = param_{old} - \\hat{g}

    Without bias correction, the equation of parameters updating is:

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*(\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}+\\lambda*param_{old})

        & param_{new} = param_{old} - \\hat{g}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-8.
        do_bias_correction (bool, optional): Whether to do the bias correction. Defaults to False.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        weight_decay (Optional[float], optional): The weight decay factor (In the equation is :math:`\\lambda`). Defaults to None.
        weight_decay_includes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that use weight decay. Defaults to None.
        weight_decay_excludes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that do not use weight decay. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    Note:

        Only one of `weight_decay_includes` and `weight_decay_excludes` can be set. If both are None,
        all the model parameters will use weight decay.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )

            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set AdamW optimizer, weight_decay factor is 0.00005
            flow.optimizer.AdamW(lr_scheduler,
                    do_bias_correction=False, weight_decay=0.00005).minimize(loss)

            return loss

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        do_bias_correction=False,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.do_bias_correction = do_bias_correction
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_adam_conf().set_beta1(self.beta1)
        optimizer_conf.mutable_adam_conf().set_beta2(self.beta2)
        optimizer_conf.mutable_adam_conf().set_epsilon(self.epsilon)
        optimizer_conf.mutable_adam_conf().set_do_bias_correction(
            self.do_bias_correction
        )
        if self.weight_decay is not None:
            optimizer_conf.mutable_weight_decay_conf().set_weight_decay_rate(
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                for weight_decay_include in self.weight_decay_includes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_includes().add_pattern(
                        weight_decay_include
                    )
            elif self.weight_decay_excludes is not None:
                for weight_decay_exclude in self.weight_decay_excludes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_excludes().add_pattern(
                        weight_decay_exclude
                    )
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class RMSProp(Optimizer):
    """The optimizer of the RMSProp algorithm.

    This algorithm uses mean squared gradient to adjust the learning rate.

    The equation of parameters updating is:

        if centered:

            .. math::

                & mg_t = mg * \\beta_1 + (1 - \\beta_1) * grad

                & denom_t = S_t - mg_t * mg_t

        else:

            .. math::

                denom_t = S_t

        .. math::

            param_{new} = param_{old} - \\frac{learning\\_rate}{\\sqrt{denom_t+\\epsilon}} \\odot grad

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        decay_rate (float, optional): The decay factor (:math:`\\beta_1`). Defaults to 0.99.
        epsilon (float, optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-8.
        centered (bool, optional): If `True`, gradients are normalized by the estimated
                                   variance of the gradient; if False, by the uncentered second moment.
                                   Setting this to `True` may help with training, but is slightly more
                                   expensive in terms of computation and memory. Defaults to `False`.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set RMSProp optimizer
            flow.optimizer.RMSProp(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        decay_rate: float = 0.99,
        epsilon: float = 1e-08,
        centered: bool = False,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.centered = centered
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_rmsprop_conf().set_decay_rate(self.decay_rate)
        optimizer_conf.mutable_rmsprop_conf().set_centered(self.centered)
        optimizer_conf.mutable_rmsprop_conf().set_epsilon(self.epsilon)
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class LARS(Optimizer):
    """The optimizer of the LARS algorithm.

    The equation of parameters updating is:

    .. math::

        & local\\_learning\\_rate = learning\\_rate*lars\\_coeff*\\frac{\\lVert{parm_{old}\\rVert}}{\\epsilon+\\lVert{grad\\rVert}+weight_decay*\\lVert{parm_{old}\\rVert}}

        & momentum_t = \\beta*momentum_{t-1} + local\\_learning\\_rate*(grad)

        & param_{new} = param_{old} - momentum_t - local_learning_rate * weight_decay * param_{old}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        momentum_beta (float, optional): The momentum factor (:math:`\\beta`). Defaults to 0.9.
        epsilon (float, optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-9.
        lars_coefficient (float, optional): The coefficient factor, it defines how much we trust the layer to change its weights (:math:`lars\\_coeff`). Defaults to 0.0001.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        weight_decay (Optional[float], optional): The weight decay factor (In the equation is :math:`\\lambda`). Defaults to None.
        weight_decay_includes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that use weight decay. Defaults to None.
        weight_decay_excludes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that do not use weight decay. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.

    Note:

        Only one of `weight_decay_includes` and `weight_decay_excludes` can be set. If both are None,
        all the model parameters will use weight decay.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
                images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
                labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.1
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            # Set LARS optimizer, momentum factor is 0.9
            flow.optimizer.LARS(lr_scheduler, momentum_beta=0.9).minimize(loss)

            return loss

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        momentum_beta: float = 0.9,
        epsilon: float = 1e-09,
        lars_coefficient: float = 0.0001,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.momentum_beta = momentum_beta
        self.epsilon = epsilon
        self.lars_coefficient = lars_coefficient
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_lars_conf().set_momentum_beta(self.momentum_beta)
        optimizer_conf.mutable_lars_conf().set_epsilon(self.epsilon)
        optimizer_conf.mutable_lars_conf().set_lars_coefficient(self.lars_coefficient)
        if self.weight_decay is not None:
            optimizer_conf.mutable_weight_decay_conf().set_weight_decay_rate(
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                for weight_decay_include in self.weight_decay_includes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_includes().add_pattern(
                        weight_decay_include
                    )
            elif self.weight_decay_excludes is not None:
                for weight_decay_exclude in self.weight_decay_excludes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_excludes().add_pattern(
                        weight_decay_exclude
                    )
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class LazyAdam(Optimizer):
    """
    The optimizer of the LazyAdam algorithm.

    This algorithm can adjust the learning rate of each parameter dynamically according to the 1st-moment estimates and the 2nd-moment estimates of the gradient.

    The difference between Adam optimizer and LazyAdam optimizer is that LazyAdam only updates the element that has gradient in the current batch, it is faster than Adam optimizer.

    .. math::

        & V_t = \\beta_1*V_{t-1} + (1-\\beta_1)*grad

        & S_t = \\beta_2*S_{t-1} + (1-\\beta_2)*{grad} \\odot {grad}

        & \\hat{g} = learning\\_rate*\\frac{{V_t}}{\\sqrt{{S_t}}+\\epsilon}

        & param_{new} = param_{old} - \\hat{g}

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-8.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    For example:

    .. code-block:: python

        import oneflow.compatible.single_client as flow
        import oneflow.compatible.single_client.typing as tp

        @flow.global_function(type="train")
        def train_job(
            images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
        ) -> tp.Numpy:
            with flow.scope.placement("gpu", "0:0"):
                logits = lenet(images, train=True)
                loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, logits, name="softmax_loss"
                )
            # Set learning rate as 0.001
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
            # Set LazyAdam optimizer
            flow.optimizer.LazyAdam(lr_scheduler).minimize(loss)

            return loss

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        loss_scale_factor: Optional[float] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_lazy_adam_conf().set_beta1(self.beta1)
        optimizer_conf.mutable_lazy_adam_conf().set_beta2(self.beta2)
        optimizer_conf.mutable_lazy_adam_conf().set_epsilon(self.epsilon)
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class LAMB(Optimizer):
    """

    Args:
        lr_scheduler (LrScheduler): The scheduler of learning rate.
        beta1 (float, optional): The exponential weighted average decay rate for the 1st-moment estimates (:math:`\\beta_1`). Defaults to 0.9.
        beta2 (float, optional): The exponential weighted average decay rate for the 2rd-moment estimates (:math:`\\beta_2`). Defaults to 0.999.
        epsilon ([type], optional): A small float constant value for numerical stability (:math:`\\epsilon`). Defaults to 1e-6.
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        weight_decay (Optional[float], optional): The weight decay factor (In the equation is :math:`\\lambda`). Defaults to None.
        weight_decay_includes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that use weight decay. Defaults to None.
        weight_decay_excludes (Optional[Union[Sequence[Text], Text]], optional): The name of the model parameters that do not use weight decay. Defaults to None.
        grad_clipping (Optional[ClipGradientConf], optional): The gradient clipping strategy. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.
        variables(Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ]): maintained variables.

    Note:

        Only one of `weight_decay_includes` and `weight_decay_excludes` can be set. If both are None,
        all the model parameters will use weight decay.

    """

    def __init__(
        self,
        lr_scheduler: LrScheduler,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-06,
        loss_scale_factor: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_includes: Optional[Union[Sequence[Text], Text]] = None,
        weight_decay_excludes: Optional[Union[Sequence[Text], Text]] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
        variables: Optional[
            Union[Sequence[Text], Callable[[], Sequence[Text]]]
        ] = GetVariablesForCurrentJob,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        if isinstance(weight_decay_includes, str):
            weight_decay_includes = [weight_decay_includes]
        if isinstance(weight_decay_excludes, str):
            weight_decay_excludes = [weight_decay_excludes]
        self.weight_decay_includes = weight_decay_includes
        self.weight_decay_excludes = weight_decay_excludes
        self.variables = variables

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        optimizer_conf = train_conf.mutable_optimizer_conf().Add()
        self.lr_scheduler.SetLrFieldsInOptimizerConf(optimizer_conf)
        if self.grad_clipping is not None:
            optimizer_conf.mutable_clip_conf().CopyFrom(self.grad_clipping.clip_conf)
        optimizer_conf.mutable_lamb_conf().set_beta1(self.beta1)
        optimizer_conf.mutable_lamb_conf().set_beta2(self.beta2)
        optimizer_conf.mutable_lamb_conf().set_epsilon(self.epsilon)
        if self.weight_decay is not None:
            optimizer_conf.mutable_weight_decay_conf().set_weight_decay_rate(
                self.weight_decay
            )
            assert not (
                self.weight_decay_excludes is not None
                and self.weight_decay_includes is not None
            )
            if self.weight_decay_includes is not None:
                for weight_decay_include in self.weight_decay_includes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_includes().add_pattern(
                        weight_decay_include
                    )
            elif self.weight_decay_excludes is not None:
                for weight_decay_exclude in self.weight_decay_excludes:
                    optimizer_conf.mutable_weight_decay_conf().mutable_excludes().add_pattern(
                        weight_decay_exclude
                    )
        for variable in self.Variables():
            optimizer_conf.add_variable_op_names(variable)


class CombinedOptimizer(Optimizer):
    """
    Combined optimizer for multi optimizer case.

    Args:
        optimizers (Sequence[Optimizer]): optimizers to work together
        loss_scale_factor (Optional[float], optional): The scale factor of loss. Defaults to None.
        train_step_lbn (Optional[Text], optional): [description]. Defaults to None.
        loss_scale_policy (Optional[LossScalePolicy]): The policy of loss scale.

        Example: see test_multi_optimizer.py
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        loss_scale_factor: Optional[float] = None,
        train_step_lbn: Optional[Text] = None,
        loss_scale_policy: Optional[LossScalePolicy] = None,
    ):
        super().__init__(loss_scale_factor, train_step_lbn, loss_scale_policy)
        for optimizer in optimizers:
            assert not isinstance(
                optimizer, CombinedOptimizer
            ), "Forbid constructing CombinedOptimizer recursively"
            assert (
                optimizer.train_step_lbn is None
            ), "Only one train step lbn among multi optimizers, please set thisparameter in CombinedOptimizer"
            assert (
                optimizer.loss_scale_policy is None
            ), "Only one loss scale policy among multi optimizers, please set thisparameter in CombinedOptimizer"
        self.optimizers = optimizers

    def Variables(self) -> List[Text]:
        if not self._variables_list_init:
            self.variables = []
            for optimizer in self.optimizers:
                self.variables.append(optimizer.Variables())
            self._variables_list_init = True
        return self.variables

    def _SanityCheck(self):
        all_variables = set(GetVariablesForCurrentJob())
        union_set = set()
        inter_set = all_variables
        for optimizer in self.optimizers:
            s = set(optimizer.Variables())
            union_set.union(s)
            inter_set = inter_set.intersection(s)
        assert union_set.issubset(all_variables)
        assert (
            len(inter_set) == 0
        ), "Do not allow overlap of variables between multi optimizers"

    def _AddOptimizerConfInTrainConf(self, train_conf) -> None:
        self._SanityCheck()
        for optimizer in self.optimizers:
            optimizer._AddOptimizerConfInTrainConf(train_conf)
