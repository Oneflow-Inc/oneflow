/*
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
*/
#include "oneflow/core/job_rewriter/pass_util.h"

namespace oneflow {

bool IsNodeInList(const HashSet<std::string>& op_list, OpNode* node) {
  if (node->op().op_conf().has_user_conf() == false) { return false; }
  const std::string op_type = node->op().op_conf().user_conf().op_type_name();
  return IsKeyFound(op_list, op_type);
}

std::string ReplaceSlashToDash4Lbn(std::string lbn) {
  std::replace(lbn.begin(), lbn.end(), '/', '-');
  return lbn;
}

void DfsTopoGraphTraversal(const OpGraph& graph, bool reversed,
                           std::function<bool(OpNode*)> IsCurNodeStartNode,
                           std::function<bool(OpNode*)> IsCurNodeSatisfied,
                           std::function<bool(OpNode*)> IsFatherNodeSatisfied,
                           std::function<void(OpNode*)> NodeHandler) {
  auto start_nodes = reversed ? graph.sink_nodes() : graph.source_nodes();
  std::function<void(OpNode*, std::function<void(OpNode*)>)> NodeOnInEdge =
      reversed ? &OpNode::ForEachNodeOnOutEdge : &OpNode::ForEachNodeOnInEdge;
  std::function<void(OpNode*, std::function<void(OpNode*)>)> NodeOnOutEdge =
      reversed ? &OpNode::ForEachNodeOnInEdge : &OpNode::ForEachNodeOnOutEdge;
  graph.DfsTopoForEachNode(start_nodes, NodeOnInEdge, NodeOnOutEdge, [&](OpNode* node) {
    if (IsCurNodeStartNode(node)) {
      NodeHandler(node);
      return;
    }
    if (IsCurNodeSatisfied(node)) {
      bool is_one_father_of_node_satisfied = false;
      NodeOnInEdge(node, [&](OpNode* father_node) {
        if (is_one_father_of_node_satisfied) { return; }
        if (IsFatherNodeSatisfied(father_node)) { is_one_father_of_node_satisfied = true; }
      });
      if (is_one_father_of_node_satisfied) { NodeHandler(node); }
    }
  });
}

}  // namespace oneflow
