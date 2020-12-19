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
from __future__ import absolute_import
from abc import ABC

from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.framework.module import Module
from oneflow.python.ops.optimizer import Optimizer


@oneflow_export("nn.Model")
class Model(
    ABC,
    Module,
):
    r"""Computation logic for a model. It can be put into all kinds of Executors(such as Trainer) to run.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # pointer to the trainer object
        self.trainer = None

    def optimizers(self):
        opts = self.trainer.optimizers

        # single optimizer
        if isinstance(opts, list) and len(opts) == 1 and isinstance(opts[0], Optimizer):
            return opts[0]
        # multiple opts
        return opts
    
    @property
    def current_epoch(self) -> int:
        return self.trainer.current_epoch if self.trainer else 0

    @property
    def automatic_optimization(self) -> bool:
        r"""If False you are responsible for calling .backward(), .step(), zero_grad().
        """
        return True
    
    def forward(self, *args, **kwargs):
        r"""Same as `nn.Module.forward()`, here is to define the operations you want to use for prediction.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Predicted output 
        """
        return super.forward(*args)
    
    def training_step(self, *args, **kwargs):
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.
        """
        raise NotImplementedError()
    
    def configure_optimizers(self):
        r""" Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        """
        raise NotImplementedError()