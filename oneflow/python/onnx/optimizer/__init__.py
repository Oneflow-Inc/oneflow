# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
"""oneflow.python.onnx.optimizer module"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import copy

from .const_fold_optimizer import ConstFoldOptimizer
from .identity_optimizer import IdentityOptimizer
from .merge_duplicated_nodes_optimizer import MergeDuplicatedNodesOptimizer
from .transpose_optimizer import TransposeOptimizer
from .loop_optimizer import LoopOptimizer
from .back_to_back_optimizer import BackToBackOptimizer
import logging

# optimizer sequence need to be considered carefully
_optimizers = OrderedDict([
    # FIXME(daquexian): transpose optimizer fails test_conv.test_conv2d_k3s1_nhwc_valid test
    ("optimize_transpose", TransposeOptimizer),
    ("fold_constants", ConstFoldOptimizer),
    ("loop_optimizer", LoopOptimizer),
    # merge_duplication should be used after optimize_transpose
    # for optimize_transpose may have some trans nodes that can be merge
    ("merge_duplication", MergeDuplicatedNodesOptimizer),
    ("remove_identity", IdentityOptimizer),
    ("remove_back_to_back", BackToBackOptimizer),
])


def _get_optimizers():
    return _optimizers


def optimize_graph(graph):
    """ Optimize graph, return optimized graph. No throw. """
    logger = logging.getLogger(__name__)
    logger.info("Optimizing ONNX model")

    before = graph.dump_node_statistics()
    opts = _get_optimizers()
    continue_flag = True
    while continue_flag:
        continue_flag = False
        for name, factory in opts.items():
            try:
                logger.debug("Apply %s", name)
                current = copy.deepcopy(graph)
                opt = factory()
                graph = opt.optimize(current) or graph
                continue_flag = continue_flag or opt.graph_been_opt

            except Exception:  # pylint: disable=broad-except
                # if current optimizer fails, continue with other optimizers
                logger.warning("Failed to apply %s", name, exc_info=1)

    try:
        graph.topological_sort(graph.get_nodes())
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed topological_sort", exc_info=1)

    after = graph.dump_node_statistics()
    diff = copy.deepcopy(after)
    diff.subtract(before)
    diff = ["{} {} ({}->{})".format(k, str(v) if v < 0 else '+' + str(v), before.get(k, 0), after.get(k, 0))
            for k, v in sorted(diff.items()) if v != 0]
    logger.info("After optimization: %s", ', '.join(diff) if diff else "no change")

    return graph
