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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_GRAPH_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_GRAPH_H_

#include <algorithm>
#include <unordered_map>
#include "oneflow/core/auto_parallel/binary_set.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_edge.h"
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace auto_parallel {

// A graph structure to deal with the SBP strategy.
// It contains a lot of eliminations to shrink the topography structure of the original graph.
// Furthermore, it contains some adjustment tricks for search a good strategy in the shrunk graph.
class SbpGraph final {
 public:
  // Constructor
  SbpGraph() = default;

  // Deconstructor
  ~SbpGraph();

  OF_DISALLOW_COPY_AND_MOVE(SbpGraph);
  bool operator==(const SbpGraph& other) { return this == &other; }

  // Randomly assign a SbpSignature strategy
  void RandomSbpSignature(bool use_sbp_collector) const;
  // assign 0 to a SbpSignature strategy to avoid randomness
  void SetDefaultSbpSig() const;

  void StoreOriginMemory();
  // Compute Cost for current strategy
  double ComputeCost() const;
  double ComputeWeightedCost() const;
  // Re-compute weighted cost
  void ReComputeWeightedCost();

  // Generate a node
  SbpNode* GenerateNode();

  // Merge all parallel edges & Check and eliminate all nodes with only one
  // degree-in and one degree-out
  int32_t NodeAndEdgeEliminations();

  // Finalize Sbp Cost for the whole graph
  void FinalizeSbp() const;

  // Use Greedy Strategy to decide Sbp for Nodes in node_list_. Should be used
  // after we have a initial strategy.
  // Set for_node to be true will only use GreedyStrategy on Nodes.
  double GreedyStrategy(bool for_node) const;
  // Use greedy strategy on the one ring neighborhood with the maximum number of points nbh_num.
  double GreedyStrategy(int32_t nbh_num = 4) const;

  // Find one strategy with finite cost for adjustment
  Maybe<void> Find1Strategy4Greedy() const;
  // Use brute force to search for a strategy with minimum cost for a neighborhood
  double NbhGreedyStrategy(std::vector<int32_t>& nbh_id2node_list_id) const;

  // Set threshold_ for SbpNode Merging
  void SetThreshold(int32_t threshold) { threshold_ = threshold; }

  // Clip an edge, remove it from graph
  // Clipping an edge will also delete the nodes and edges contained in this edge. Though not
  // suffering from any compiling and runtime bugs, clipping an edge on a shrunk graph is not
  // recommended. We should carefully think about it before any clipping.
  void ClipEdge(SbpEdge* this_edge) const;

  // Compute the minimum and maximum layer of each node in the graph
  int32_t ComputeLayer(
      HashMap<std::string, SbpNode*>& op_name2sbp_node,
      const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps) const;

  // Find the trunk of the sbp graph, then reduce the wait time for tributaries
  void FindTrunk(int32_t max_min_layer, HashMap<std::string, SbpNode*>& op_name2sbp_node) const;

  // Set wait time
  void SetWaitTime(double wait_time);

  // Constant getter
  std::vector<SbpNode*>& GetNodeList() { return node_list_; }
  int64_t GetMemory() const;

 private:
  friend class SbpCollector;
  friend class SbpConstructor;

  // All the nodes
  std::vector<SbpNode*> node_list_;

  // Limitation: Merged node should not have a number of Sbp Signature greater
  // than threshold.
  int32_t threshold_ = 100;
  // Wait time for copy cost, which occurs before communication between devices.
  double wait_time_ = 16500.0;

  // Remove a node from the node list
  void RemoveFromNodeList(SbpNode* this_node);

  // Check and eliminate one node with only one degree-in and one degree-out
  int32_t NodeElimination(SbpNode* this_node);
  // Merge all parallel edges with given start_node_ and end_node_
  int32_t EdgeElimination(SbpNode* this_node) const;
  // Check and eliminate one child node
  int32_t ChildElimination(SbpNode* this_node);

  // Merge two nodes
  int32_t NodeMerging(SbpNode* first, SbpNode* second);
  // Select two nodes and merge them
  int32_t PickAndMerge();

  void DfsAddNbhCost(std::vector<int32_t>& nbh_id2node_list_id,
                     std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                     std::vector<int32_t>& order2nbh_id, std::vector<int32_t>& nbh_id2order,
                     std::vector<double>& order2acc_min_in_nbh_cost,
                     std::vector<std::vector<double>>& out_nbh_costs,
                     std::vector<std::vector<int32_t>>& nbh_id2order2sbp_id,
                     std::vector<int32_t>& min_sbp_sig_id, double& min_cost, int32_t order,
                     double curr_cost) const;

  bool DfsFindReasonableCost(std::vector<int32_t>& nbh_id2node_list_id,
                             std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                             std::vector<int32_t>& nbh_id2order, int32_t nbh_id) const;
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_GRAPH_H_
