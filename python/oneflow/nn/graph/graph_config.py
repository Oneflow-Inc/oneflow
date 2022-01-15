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
        self._outputs_buffer_size = 2
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

    def set_outputs_buffer_size(self, value: int = 2):
        r"""Set the outputs buffer size of ``nn.Graph``.
        When graph's outputs buffer size is greater than 2, multiple call on the
        graph can work like a pipeline. This makes multiple call takes less time.
        
        The default outputs buffer size is 2.

        Args:
            value (int): graph ouputs buffer size.
        """
        self._outputs_buffer_size = value

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
            value (int): num of steps.
        """
        self.proto.set_num_gradient_accumulation_steps(value)

    def set_zero_redundancy_optimizer_mode(self, mode: str = "distributed_split"):
        """Set mode to remove redundancy of optimizer states.
        This optimzation will reduce optimizer states memory consumption as described
        by ZeRO https://arxiv.org/abs/1910.02054 .


        Args:
            mode (str): "distributed_split" or "non_distributed". "distributed_split" mode
                         will shard each optimizer state across devices. "non_distributed" mode
                         will place each optimizer state to only one device.
        """
        assert mode in ("distributed_split", "non_distributed")
        self.proto.set_optimizer_placement_optimization_mode(mode)

    def set_zero_redundancy_optimizer_min_size_after_split(self, value):
        """Set the min size of optimizer state/grad/parameter after split.

        Args:
            value (int): min size value.
        """
        assert isinstance(value, int)
        assert value >= 1
        self.proto.set_optimizer_placement_optimization_threshold(value)

    def enable_xla_jit(self, value=True):
        """Whether use xla_jit in xrt or not. When this option enable, oneflow will check all operators is supported by 
           xla_jit or not. Clustering supported operators as subgraph, then runing subgraph by xla_jit.

           XLA: https://www.tensorflow.org/xla

        Args:
            value (bool, optional): [description]. Defaults to True.
        """
        self.proto.mutable_xrt_config().set_use_xla_jit(value)

    def enable_tensorrt(self, value=True):
        """Whether use tensorrt in xrt or not. When this option enable, oneflow will check all operators is supported by 
           tensorrt or not. Clustering supported operators as subgraph, then runing subgraph by tensorrt.

           TensorRT: https://developer.nvidia.com/tensorrt

        Args:
            value (bool, optional): [description]. Defaults to True.
        """
        self.proto.mutable_xrt_config().set_use_tensorrt(value)

    def enable_openvino(self, value=True):
        """Whether use openvino in xrt or not. When this option enable, oneflow will check all operators is supported by 
           openvino or not. Clustering supported operators as subgraph, then runing subgraph by openvino.

           Please note that, openvino only support inference mode.

           OpenVINO: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html

        Args:
            value (bool, optional): [description]. Defaults to True.
        """
        self.proto.mutable_xrt_config().set_use_openvino(value)

    def enable_cudnn_conv_heuristic_search_algo(self, mode: bool = True):
        """ Whether enable cudnn conv operatioin to use heuristic search algorithm.
    
        Args:
            mode (bool, optional): Whether enable cudnn conv operatioin to use heuristic
                                   search algorithm. Default is True.
        """
        self.proto.set_cudnn_conv_heuristic_search_algo(mode)

    def _generate_optimizer_and_variable_configs(
        self, opt_dict: OptDict = None, variables_conf: OrderedDict = None,
    ):
        opt_dict.generate_optimizer_and_variable_configs(self.proto, variables_conf)

    def __repr__(self):
        main_str = (
            "("
            + "CONFIG"
            + ":config:"
            + self.__class__.__name__
            + "("
            + ("training=" + str(self.training) + ", ")
            + "))"
        )
        return main_str
