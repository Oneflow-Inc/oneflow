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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/parallel_desc.h"

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

std::function<bool(const OpNode* op_node)> MakePredicatorIsSafeToDelete(const OpGraph& op_graph) {
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });
  return [=](const OpNode* op_node) {
    if (op_node->out_edges().size() > 1) { return false; }
    if (!op_node->op().op_conf().ctrl_in_op_name().empty()) { return false; }
    if (ctrl_in_op_names.find(op_node->op().op_conf().name()) != ctrl_in_op_names.end()) {
      return false;
    }
    return true;
  };
}

bool IsUserOpWithTypeName(const OperatorConf& op_conf, const std::string& op_type_name) {
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == op_type_name;
}

std::string GenParallelConfKey(const ParallelConf& conf) {
  std::string ret = conf.device_tag();
  for (const auto& name : conf.device_name()) { ret += ("-" + name); }
  return ret;
}

void InsertCtrlEdgeInChain(const std::vector<const OpNode*>& ordered_op_nodes,
                           std::function<bool(const std::string&, const std::string&)>& IsReachable,
                           HashMap<std::string, OperatorConf>* mut_op_name2conf) {
  HashMap<std::string, const OpNode*> placement2op_node;

  for (int64_t i = 0; i < ordered_op_nodes.size(); ++i) {
    const OpNode* this_node = CHECK_JUST(VectorAt(ordered_op_nodes, i));
    const auto& this_op_conf = this_node->op().op_conf();
    if (this_op_conf.has_src_subset_tick_conf() || this_op_conf.has_dst_subset_tick_conf()) {
      continue;
    }
    auto key = GenParallelConfKey(this_node->parallel_desc().parallel_conf());
    auto it_placement7op_node = placement2op_node.find(key);
    if (it_placement7op_node == placement2op_node.end()) {
      // Update previous op
      placement2op_node[key] = this_node;
    } else {
      // const OpNode* prev_node = CHECK_JUST(VectorAt(ordered_op_nodes, i - 1));
      auto& prev_node = it_placement7op_node->second;
      const std::string& this_op_name = this_node->op().op_name();
      const std::string& prev_op_name = prev_node->op().op_name();
      // If there exist a path from the source node to the target node,
      // then we do not need to add the control edge since the target node is already controlled.
      // If there exist a path from the target node to the source node,
      // then we can not add the control edge since it will cyclize them.
      // a -> ... -> b -> c -> a
      if (!(IsReachable(prev_op_name, this_op_name) || IsReachable(this_op_name, prev_op_name))) {
        auto it = mut_op_name2conf->find(this_op_name);
        // If this op have not been modified, put it in the map.
        if (it == mut_op_name2conf->end()) {
          it = mut_op_name2conf->emplace(this_op_name, this_op_conf).first;
        }
        it->second.add_ctrl_in_op_name(prev_op_name);
      }

      // Update previous op
      prev_node = this_node;
    }
  }
};

}  // namespace oneflow
