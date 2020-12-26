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
import oneflow.core.summary.projector_pb2 as projector_pb2
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api
import time
import logging

import oneflow as flow


@oneflow_export("summary.Graph")
class Graph(object):
    r"""The class of Graph

    This class can write 'computing_graph' or 'structure_graph' into log file
    """

    def __init__(self, logdir=None):
        r"""Create a Graph object

        Args:
            logdir: The log dir

        Raises:
            Exception: If log dir is None or illegal
        """
        if logdir is None:
            raise Exception("logdir should not be None!")
        logdir += "/graph"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.logdir_ = logdir
        self.structure_graph_filename_ = None
        self.compute_graph_filename_ = None

    def write_structure_graph(self):
        if (self.structure_graph_filename_ is not None) and (
            os.path.exists(self.structure_graph_filename_)
        ):
            raise OSError("You must create only one structure graph log file!")

        self.structure_graph_filename_ = self.logdir_ + "/structure_graph.json"
        struct_graph_str = oneflow_api.GetStructureGraph()
        with open(self.structure_graph_filename_, "w", encoding="utf-8") as f:
            f.write(str(struct_graph_str))
            f.flush()

    @property
    def logdir(self):
        return self.logdir_

    @property
    def structure_graph_filename(self):
        return self.structure_graph_filename_
