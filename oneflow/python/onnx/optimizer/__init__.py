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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# oneflow.python.onnx.optimizer module

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
_optimizers = OrderedDict(
    [
        ("optimize_transpose", TransposeOptimizer),
        ("fold_constants", ConstFoldOptimizer),
        ("loop_optimizer", LoopOptimizer),
        # merge_duplication should be used after optimize_transpose
        # for optimize_transpose may have some trans nodes that can be merge
        ("merge_duplication", MergeDuplicatedNodesOptimizer),
        ("remove_identity", IdentityOptimizer),
        ("remove_back_to_back", BackToBackOptimizer),
    ]
)


def _get_optimizers():
    return _optimizers


def OptimizeGraph(graph):
    """ Optimize graph, return optimized graph. No throw. """
    logger = logging.getLogger(__name__)
    logger.info("Optimizing ONNX model")

    before = graph.DumpNodeStatistics()
    opts = _get_optimizers()
    continue_flag = True
    while continue_flag:
        continue_flag = False
        for name, factory in opts.items():
            try:
                logger.debug("Apply %s", name)
                current = copy.deepcopy(graph)
                opt = factory()
                graph = opt.Optimize(current) or graph
                continue_flag = continue_flag or opt.graph_been_opt

            except Exception:  # pylint: disable=broad-except
                # if current optimizer fails, continue with other optimizers
                logger.warning("Failed to apply %s", name, exc_info=1)

    try:
        graph.TopologicalSort(graph.get_nodes())
    except Exception:  # pylint: disable=broad-except
        logger.warning("Failed TopologicalSort", exc_info=1)

    after = graph.DumpNodeStatistics()
    diff = copy.deepcopy(after)
    diff.subtract(before)
    diff = [
        "{} {} ({}->{})".format(
            k, str(v) if v < 0 else "+" + str(v), before.get(k, 0), after.get(k, 0)
        )
        for k, v in sorted(diff.items())
        if v != 0
    ]
    logger.info("After optimization: %s", ", ".join(diff) if diff else "no change")

    return graph
