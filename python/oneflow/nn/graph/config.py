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
from collections import OrderedDict

from oneflow.nn.graph.optimizer import OptDict
import oneflow._oneflow_internal.oneflow.core.job.job_conf as job_conf_cfg


class GraphConfig(object):
    r"""For configuration of nn.Graph.
    """

    def __init__(self):
        super().__init__()
        self.proto = job_conf_cfg.JobConfigProto()
        self._train(False)

    def _train(self, mode: bool = True):
        if mode:
            self.proto.mutable_train_conf()
        else:
            self.proto.mutable_predict_conf()

    @property
    def training(self):
        if self.proto.has_train_conf():
            return True
        if self.proto.has_predict_conf():
            return False
        raise NotImplementedError

    def enable_amp(self, mode: bool = True):
        """If true, then graph will use mixed precision mode, it means use both float16 and float32 during model training.

        Args:
            mode (bool, optional): [description]. Default is True.
        """
        assert type(mode) is bool
        self.proto.set_enable_auto_mixed_precision(mode)

    def allow_fuse_model_update_ops(self, mode: bool = True):
        """If true, try to fuse cast + scale + l1_l2_regularize_gradient + model_update to one op to improve performance.

        Args:
            mode (bool, optional): [description]. Default is True.
        """
        self.proto.set_enable_fuse_model_update_ops(mode)

    def allow_fuse_add_to_output(self, mode: bool = True):
        """If true, try to fuse a binary element-wise add to one of the predecessors to improve performance.

        Args:
            mode (bool, optional): [description]. Default is True.
        """
        self.proto.set_enable_fuse_add_to_output(mode)

    def allow_fuse_cast_scale(self, mode: bool = True):
        """If true, try to fuse cast and scalar_mul_by_tensor to improve performance.
    
        Args:
            mode (bool, optional): [description]. Default is True.
        """
        self.proto.set_enable_fuse_cast_scale(mode)

    def set_gradient_accumulation_steps(self, value):
        """Set num of steps to accumulate gradient.

        Args:
            value (int): num of steps
        """
        self.proto.set_num_gradient_accumulation_steps(value)

    def _generate_optimizer_and_variable_configs(
        self, opt_dict: OptDict = None, variables_conf: OrderedDict = None,
    ):
        opt_dict.generate_optimizer_and_variable_configs(
            self.proto.mutable_train_conf(), variables_conf
        )
