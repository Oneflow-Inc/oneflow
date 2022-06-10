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
import os

from collections import OrderedDict

from oneflow.nn.graph.optimizer import OptDict
import oneflow.core.job.job_conf_pb2 as job_conf_pb


class GraphConfig(object):
    r"""For configuration of nn.Graph.
    """

    def __init__(self):
        super().__init__()
        self._outputs_buffer_size = 2
        self.proto = job_conf_pb.JobConfigProto()
        self._train(False)

    def _train(self, mode: bool = True):
        if mode:
            self.proto.train_conf.SetInParent()
        else:
            self.proto.predict_conf.SetInParent()

    @property
    def training(self):
        if self.proto.HasField("train_conf"):
            return True
        if self.proto.HasField("predict_conf"):
            return False
        raise NotImplementedError

    def set_outputs_buffer_size(self, value: int = 2):
        r"""Set the outputs buffer size of ``nn.Graph``.

        When graph's outputs buffer size is greater than 2, multiple call on the graph can work like a pipeline. This makes multiple call takes less time.

        The default outputs buffer size is 2.

        # TODO (lixiang): Explain the meaning of the size of buffer size and add sample code.
        # The size of the buffer size indicates the maximum number of iterations that the output of the Graph and the Graph actually executed asynchronously can overlap.
        # If the buffer size is 1, there is no pipeline. A size of 2 means that it can execute 1 iter ahead of time. A size of 3 means that two iters can be executed ahead of time.

        Args:
            value (int): graph ouputs buffer size.
        """
        self._outputs_buffer_size = value

    def enable_amp(self, mode: bool = True):
        r"""If set to true, then graph will use mixed precision mode, it means use both float16 and float32 during model training.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.enable_amp(True) # Use mixed precision mode.
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            mode (bool, optional): The default vaule is True.
        """
        assert type(mode) is bool
        self.proto.enable_auto_mixed_precision = mode

    def allow_fuse_model_update_ops(self, mode: bool = True):
        r"""If set to true, try to fuse cast + scale + l1_l2_regularize_gradient + model_update to one op to improve performance.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.allow_fuse_model_update_ops(True)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            mode (bool, optional): The default vaule is True.
        """
        self.proto.enable_fuse_model_update_ops = mode

    def allow_fuse_add_to_output(self, mode: bool = True):
        r"""If set to true, try to fuse a binary element-wise add operetor to one of the predecessors to improve performance.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.bn1 = flow.nn.BatchNorm1d(100)
                    self.config.allow_fuse_add_to_output(True)
                def build(self, x):
                    bn = self.bn1(x) 
                    out = bn + x
                    return out

            graph = Graph()

        Args:
            mode (bool, optional): The default vaule is True.
        """
        self.proto.enable_fuse_add_to_output = mode

    def allow_fuse_cast_scale(self, mode: bool = True):
        r"""If set to true, try to fuse cast and scalar_mul_by_tensor to improve performance.
    
        For example:

        .. code-block:: python

            import oneflow as flow

            def model(x):
                return flow.mul(1,flow.cast(x,flow.int8))

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.m=model
                    self.config.allow_fuse_cast_scale(True)
                def build(self, x):
                    return self.m(x)

            graph = Graph()

        Args:
            mode (bool, optional): The default vaule is True.
        """
        self.proto.enable_fuse_cast_scale = mode

    def set_gradient_accumulation_steps(self, value):
        r"""Set num of steps to accumulate gradient.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    # Let graph do gradient accumulation, such as pipelining parallelism depends on gradient accumulation.
                    self.config.set_gradient_accumulation_steps(4)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            value (int): num of steps.
        """
        self.proto.num_gradient_accumulation_steps = value

    def set_zero_redundancy_optimizer_mode(self, mode: str = "distributed_split"):
        r"""Set mode to remove redundancy of optimizer states.
        This optimzation will reduce optimizer states memory consumption as described
        by ZeRO https://arxiv.org/abs/1910.02054 .

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            mode (str): "distributed_split" or "non_distributed". "distributed_split" mode
                         will shard each optimizer state across devices. "non_distributed" mode
                         will place each optimizer state to only one device.
        """
        assert mode in ("distributed_split", "non_distributed")
        self.proto.optimizer_placement_optimization_mode = mode

    def set_zero_redundancy_optimizer_min_size_after_split(self, value):
        r"""Set the min size of optimizer state/grad/parameter after split.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear = flow.nn.Linear(3, 8, False)
                    self.config.set_zero_redundancy_optimizer_mode("distributed_split")
                    self.config.set_zero_redundancy_optimizer_min_size_after_split(1)
                def build(self, x):
                    return self.linear(x)

            graph = Graph()

        Args:
            value (int): min size value.
        """
        assert isinstance(value, int)
        assert value >= 1
        self.proto.optimizer_placement_optimization_threshold = value

    def enable_cudnn_conv_heuristic_search_algo(self, mode: bool = True):
        r""" Whether enable cudnn conv operatioin to use heuristic search algorithm.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.m = flow.nn.Conv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                    # Do not enable the cudnn conv operation to use the heuristic search algorithm.
                    self.config.enable_cudnn_conv_heuristic_search_algo(False)
                def build(self, x):
                    return self.m(x)

            graph = Graph()
    
        Args:
            mode (bool, optional): The default vaule is True.
        """
        self.proto.cudnn_conv_heuristic_search_algo = mode

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
