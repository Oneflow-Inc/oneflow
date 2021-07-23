import logging
import os
import time

import oneflow as flow
import oneflow._oneflow_internal
import oneflow.core.summary.projector_pb2 as projector_pb2


class Graph(object):
    """The class of Graph

    This class can write 'computing_graph' or 'structure_graph' into log file
    """

    def __init__(self, logdir=None):
        """Create a Graph object

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
        if self.structure_graph_filename_ is not None and os.path.exists(
            self.structure_graph_filename_
        ):
            raise OSError("You must create only one structure graph log file!")
        self.structure_graph_filename_ = self.logdir_ + "/structure_graph.json"
        struct_graph_str = oneflow._oneflow_internal.GetSerializedStructureGraph()
        with open(self.structure_graph_filename_, "w", encoding="utf-8") as f:
            f.write(str(struct_graph_str))
            f.flush()

    @property
    def logdir(self):
        return self.logdir_

    @property
    def structure_graph_filename(self):
        return self.structure_graph_filename_
