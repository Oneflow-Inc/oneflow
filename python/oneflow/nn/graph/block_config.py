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


class BlockConfig(object):
    r"""Configurations on Module Block in nn.Graph.

    When an nn.Module is added into an nn.Graph, it is wrapped into a ModuleBlock. You can set or get optimization configs on an nn.Module with it's `ModuleBlock.config`.
    """

    def __init__(self):
        self._is_null = True
        self._stage_id = None
        self._stage_placement = None
        self._activation_checkpointing = None

    # NOTE(lixiang): For the normal display of docstr, the API Doc of the get and set methods are written together in the stage_id function.
    @property
    def stage_id(self):
        r"""Set/Get stage id of nn.Module/ModuleBlock in pipeline parallelism.

        When calling stage_id(value: int = None), set different module's stage id to hint the graph
        preparing right num of buffers in pipeline. (Not Recommended, for easy and efficient pipeline
        parallelism experience, please use config.set_stage(stage_id, placement))
        """
        return self._stage_id

    @stage_id.setter
    def stage_id(self, value: int = None):
        r"""Set stage id of Block in pipeline parallelism.
        Set different module's stage id to hint the graph preparing right num of buffers in pipeline.
        """
        print(
            "Warning: `config.stage_id = i` is deprecated, please use \n",
            " config.set_stage(i, placement) for easy and efficient Pipeline parallel experience.",
        )

        self._is_null = False
        self._stage_id = value

    def set_stage(self, stage_id: int = None, placement=None):
        r"""Set stage id and placement of nn.Module/ModuleBlock in pipeline parallelism.

        Args:
            stage_id (int): stage id of this module.
            placement (flow.placement): the placement of all tensor in this module.

        Note:
            There will be automatically do tensor.to_global(placement) for all input tensor of
            this module. So there is no need to write to_global() in the module forward when using
            Pipeline Parallelism which is not recommended.

        For example:

        .. code-block:: python

            # m_stage0 and m_stage1 are the two pipeline stages of the network, respectively.
            # We can set Stage ID and Placement by using Module.config.set_stage()
            # The Stage ID is numbered starting from 0 and increasing by 1.
            # The Placement is all tensors placement of this module.
            P_0 = flow.placement(type = "cuda", ranks = [0, 1])
            P_1 = flow.placement(type = "cuda", ranks = [2, 3])
            self.module_pipeline.m_stage0.config.set_stage(stage_id = 0, placement = P0)
            self.module_pipeline.m_stage1.config.set_stage(stage_id = 1, placement = P1)

        """

        self._is_null = False
        self._stage_id = stage_id
        self._stage_placement = placement

    # NOTE(lixiang): For the normal display of docstr, the API Doc of the get and set methods are written together in the activation_checkpointing function.
    @property
    def activation_checkpointing(self):
        r"""Set/Get whether do activation checkpointing in this nn.Module.

        For example:

        .. code-block:: python

            import oneflow as flow

            class Graph(flow.nn.Graph):
                def __init__(self):
                    super().__init__()
                    self.linear1 = flow.nn.Linear(3, 5, False)
                    self.linear2 = flow.nn.Linear(5, 8, False)
                    self.linear1.config.activation_checkpointing = True
                    self.linear2.config.activation_checkpointing = True

                def build(self, x):
                    y_pred = self.linear1(x)
                    y_pred = self.linear2(y_pred)
                    return y_pred

            graph = Graph()

        """
        return self._activation_checkpointing

    @activation_checkpointing.setter
    def activation_checkpointing(self, value: bool = False):
        r"""Set whether do activation checkpointing in this Block.
        """
        self._is_null = False
        self._activation_checkpointing = value

    def __repr__(self):
        main_str = (
            "("
            + "CONFIG"
            + ":config:"
            + self.__class__.__name__
            + "("
            + (
                ("stage_id=" + str(self.stage_id) + ", ")
                if self.stage_id is not None
                else ""
            )
            + (
                (
                    "activation_checkpointing="
                    + str(self.activation_checkpointing)
                    + ", "
                )
                if self.activation_checkpointing is not None
                else ""
            )
            + "))"
        )
        return main_str
