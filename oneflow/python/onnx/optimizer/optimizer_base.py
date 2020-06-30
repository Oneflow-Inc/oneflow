# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Graph optimizer Base"""

from __future__ import unicode_literals

import copy
import logging

from .. import util


class GraphOptimizerBase(object):
    """optimizer graph to improve performance
    """

    def __init__(self):
        self._logger = logging.getLogger(
            ".".join(__name__.split(".")[:-1] + [self.__class__.__name__])
        )
        self._graph_been_opt = False

    @property
    def logger(self):
        return self._logger

    @property
    def graph_been_opt(self):
        return self._graph_been_opt

    @graph_been_opt.setter
    def graph_been_opt(self, value):
        self._graph_been_opt = value

    def Optimize(self, graph):
        """ optimize graph, return optimized graph. """
        before = graph.DumpNodeStatistics()

        graph = self._Optimize(graph)
        graph.UpdateProto()
        graph.DeleteUnusedNodes(graph.outputs)

        after = graph.DumpNodeStatistics()
        self._PrintStatDiff(before, after)
        return graph

    def _Optimize(self, graph):
        """ Derived class should override this function. """
        raise NotImplementedError

    @staticmethod
    def _ApplyOptimization(graph, optimize_func):
        """
        optimize graph
        will also optimize graph of nodes'
        Args:
            graph: the top level graph to be optimized
            optimize_func: function to optimize graph
        """
        graph = optimize_func(graph)
        for node in graph.get_nodes():
            body_graphs = node.get_body_graphs()
            if body_graphs:
                for attr, b_g in body_graphs.items():
                    b_g = GraphOptimizerBase._ApplyOptimization(b_g, optimize_func)
                    node.set_body_graph_as_attr(attr, b_g)
        return graph

    def _PrintStatDiff(self, before, after):
        diff = copy.deepcopy(after)
        diff.subtract(before)
        diff = [
            "{} {} ({}->{})".format(
                k, str(v) if v < 0 else "+" + str(v), before.get(k, 0), after.get(k, 0)
            )
            for k, v in sorted(diff.items())
            if v != 0
        ]
        self.logger.debug(", ".join(diff) if diff else "no change")
