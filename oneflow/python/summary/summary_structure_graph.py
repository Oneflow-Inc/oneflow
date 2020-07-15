import os
import oneflow.customized.utils.projector_pb2 as projector_pb2
import oneflow.python.framework.c_api_util as c_api_util
from oneflow.python.oneflow_export import oneflow_export
import time
import logging

import oneflow as flow


@oneflow_export("summary.Graph")
class Graph(object):
    def __init__(self, logdir=None):
        if logdir is None:
            raise Exception("logdir should not be None!")
        if not os.path.isdir(logdir):
            raise Exception("Logdir %r is illegal!" % logdir)
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
        struct_graph_str = c_api_util.GetStructureGraph()
        with open(self.structure_graph_filename_, "w", encoding="utf-8") as f:
            f.write(str(struct_graph_str))
            f.flush()

    @property
    def logdir(self):
        return self.logdir_

    @property
    def structure_graph_filename(self):
        return self.structure_graph_filename_
