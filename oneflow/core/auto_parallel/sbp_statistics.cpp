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

#include "oneflow/core/auto_parallel/sbp_statistics.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {
namespace auto_parallel {

void SbpStatistics::CollectStatistics(const SbpGraph<cfg::NdSbpSignature>& sbp_graph) {
  for (const auto* this_node : sbp_graph.NodeList) {
    CollectStatistics(*this_node);
    // Collect statistics for the output edges
    for (const auto* edge_out : this_node->EdgesOut) { CollectStatistics(*edge_out); }
  }
}

void SbpStatistics::CollectStatistics(const SbpNode<cfg::NdSbpSignature>& sbp_node) {
  // Collect statistics for this single node
  if (sbp_node.op_node) {
    op_num_++;
    total_cost_ += sbp_node.GetCurrCost();
    total_comp_cost_ += sbp_node.GetCurrCost();
  }

  // Collect statistics for children
  for (int32_t i = 0; i < sbp_node.Children.size(); i++) {
    CollectStatistics(*sbp_node.Children[i]);
    for (const auto* edge_in : sbp_node.Children[i]->EdgesIn) { CollectStatistics(*edge_in); }
    for (const auto* edge_out : sbp_node.Children[i]->EdgesOut) { CollectStatistics(*edge_out); }
  }

  // Collect statistics for the merged node
  if (!sbp_node.HalfNode.empty()) {
    for (const auto* half_node : sbp_node.HalfNode) {
      CollectStatistics(*half_node);
      // Collect the only edge between the two half nodes if exists
      for (const auto* edge_out : half_node->EdgesOut) { CollectStatistics(*edge_out); }
    }
  }
}

void SbpStatistics::CollectStatistics(const SbpEdge<cfg::NdSbpSignature>& sbp_edge) {
  if (sbp_edge.EdgeList.empty()) {
    // Collect statistics for this single edge
    double curr_cost = sbp_edge.GetCurrCost();
    if (curr_cost > 0) {
      total_cost_ += curr_cost;
      total_copy_cost_ += curr_cost;
      num_comm_++;
      if (curr_cost < 1.65e5) { num_slight_comm_++; }
    }
  } else {
    // Collect statistics for the middle node
    if (sbp_edge.MidNode) { CollectStatistics(*sbp_edge.MidNode); }
    // Collect statistics for all the contained edges
    for (const auto* this_edge : sbp_edge.EdgeList) { CollectStatistics(*this_edge); }
  }
}

void SbpStatistics::PrintStatistics() {
  std::cout << "Number of operators: " << op_num_ << std::endl;
  std::cout << "Total cost: " << total_cost_ << std::endl;
}

}  // namespace auto_parallel

}  // namespace oneflow