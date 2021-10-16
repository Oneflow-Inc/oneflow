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
    r"""Configurations on Block in nn.Graph.
    """

    def __init__(self):
        self._is_null = True
        self._stage_id = None
        self._activation_checkpointing = None

    @property
    def stage_id(self):
        r"""Get stage id of Block in pipeline parallelism.
        """
        return self._stage_id

    @stage_id.setter
    def stage_id(self, value: int = None):
        r"""Set stage id of Block in pipeline parallelism.
        Set different module's stage id to hint the graph preparing right num of buffers in pipeline.
        """
        self._is_null = False
        self._stage_id = value

    @property
    def activation_checkpointing(self):
        r"""Get whether do activation checkpointing in this Block.
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
