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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_NODE_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_NODE_H_

#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include "oneflow/core/auto_parallel/binary_set.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/auto_parallel/algorithm_util.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {
namespace auto_parallel {

class SbpEdge;

// A node structure to deal with the SBP strategy.
// Please see SbpGraph for the whole algorithm and introduction.
class SbpNode final {
 public:
  // default constructor
  SbpNode() : final_sbp_sig_id_(0) {}

  // This constructor is to merge two node into one
  SbpNode(SbpNode* first, SbpNode* second);

  ~SbpNode();

  OF_DISALLOW_COPY_AND_MOVE(SbpNode);
  bool operator==(const SbpNode& other) { return this == &other; }

  // another node point to this node
  void PointFrom(SbpNode* start_node);
  // this node point to another node
  void PointTo(SbpNode* end_node);

  SbpEdge* FindEdgeWithNode(const SbpNode* other_node) const;

  // Check and eliminate one child node.
  // Only used by SbpGraph since it need to remove it from the NodeList after this.
  bool EliminateItselfAsChild();

  // Initialize SbpSignature from Signature Objects
  void InitializeSbp();
  // Decide to use this SbpSignature
  const NdSbpSignature& FinalSbpSignature() const;

  // Recompute Computation Cost after adding child nodes in it
  void SummarizeCost();
  // Compute the weighted sum of the time and memory cost
  void ComputeWeightedCost();
  // Generate the relationship between this merged node and its components
  void GenerateComponentRelationship();
  // Determine Final SbpSignature for attachment of this node
  void FinalizeSbp();
  // Use Greedy Strategy to pick the sbp signature with minimum cost for this
  // node You should have an initial strategy before running this
  double GreedyStrategy();
  // Evaluate summery of cost between neighborhood and outside nodes
  double EvalOutNbhCost(const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id) const;
  // Evaluate summery of cost within neighborhood
  // We only accumulate the edge cost with a lower order.
  double EvalInNbhCost(const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                       const std::vector<int32_t>& nbh_id2order) const;
  // Evaluate summery of cost within neighborhood
  // We only accumulate the minimum edge cost with a higher order.
  double EvalMinInNbhCost(const std::unordered_map<int32_t, int32_t>& node_list_id2nbh_id,
                          const std::vector<int32_t>& nbh_id2order) const;
  // Get the one ring neighborhood of this node, which is itself and all the adjacent nodes.
  void OneRingNeighborhood(std::vector<int32_t>& nbh_1ring) const;
  // Get the n ring neighborhood of this node
  // Pre-allocate buffer, which will be faster.
  void NRingNeighborhood(int32_t n, std::vector<int32_t>& nbh_n_ring,
                         std::vector<int32_t>& nbh_1ring, const std::vector<SbpNode*>& node_list,
                         std::vector<bool>& node_tags) const;

  // Get or compute the minimum layer of this node
  int32_t GetMinLayer(
      const HashMap<std::string, SbpNode*>& op_name2sbp_node,
      const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps);
  // Spread the minimum layer to compute the maximum layer of producers
  void SpreadMaxLayer(
      const HashMap<std::string, SbpNode*>& op_name2sbp_node,
      const HashMap<const OpNode*, HashSet<std::string>>& op_node2mutable_op_ctrl_deps);
  // Set max_layer_ = min_layer_ if this node does not have any consumer
  void LiftMaxLayer();
  // Set max_layer_ = upper_bound if this node does not have any consumer
  void LiftMaxLayer(int32_t upper_bound);
  // Compute maximum layer for tributaries
  void SpreadTributaryLayer(const HashMap<std::string, SbpNode*>& op_name2sbp_node);
  // Drop down the tributary layer
  void DropTributaryLayer(int32_t upper_bound);

  // Get the minimum element in Cost
  double GetMinCost() const;
  // get the cut ratio
  double GetCutRatio() const;

  // Judge if this node is on the trunk
  // If so, judge it for its producer/upstream nodes
  void SpreadTrunk(const HashMap<std::string, SbpNode*>& op_name2sbp_node);
  // Count consumers and any downstream nodes defined by control edges
  // for producers or upstream nodes
  void RaiseConsumerNum(const HashMap<std::string, SbpNode*>& op_name2sbp_node);
  // Compute the minimal available wait time for producers or upstream nodes
  void SpreadAvailWaitTime(const std::vector<double>& trunk_cost,
                           const std::vector<double>& acc_trunk_cost,
                           const HashMap<std::string, SbpNode*>& op_name2sbp_node,
                           double wait_time);
  // Reduce and set the wait time for op in the trunk
  void SetTrunkWaitTime(double trunk_wait_time);

  // Assemble copy cost and partial memory cost for all the incoming edges
  void InitCopyAndMemoryCost(bool use_sbp_collector, bool nccl_not_use_compute_stream);
  // Assemble memory cost
  void InitializeMemory(bool is_reusable, const HashMap<LogicalBlobId, int32_t>& lbi2id,
                        const std::vector<int32_t>& id2count, bool nccl_use_compute_stream);

  // Constant getter
  int32_t GetMinLayer() const { return min_layer_; }
  int32_t GetTributaryLayer() const { return tributary_layer_; }
  OpNode* GetOperatorNode() const { return op_node_; }
  const std::vector<SbpEdge*>& GetEdgesIn() const { return edges_in_; }
  const std::vector<SbpEdge*>& GetEdgesOut() const { return edges_out_; }
  int64_t GetMemory(int32_t i) const { return in_memory_support_ ? memory_[i] : 0; }
  // Get the current memory with the current sbp signature index
  int64_t GetMemory() const { return GetMemory(final_sbp_sig_id_); }
  double GetWeightedCost(int32_t i) const { return weighted_cost_[i]; }
  // Get the current weighted cost with the current sbp signature index
  double GetWeightedCost() const { return GetWeightedCost(final_sbp_sig_id_); }
  int32_t GetComponentSbpId(int32_t merged_id, SbpNode* component_node) const;
  // Judge if sbp_node is a port of the current node
  bool IsComponent(SbpNode* sbp_node) const;

  // Setter
  void SetInMemorySupport(bool in_memory_support) { in_memory_support_ = in_memory_support; }

 private:
  friend class SbpEdge;
  friend class SbpGraph;
  friend class SbpCollector;
  friend class SbpConstructor;

  // compound edge in
  std::vector<SbpEdge*> edges_in_;
  // compound edge out
  std::vector<SbpEdge*> edges_out_;

  // Location in node_list of SbpGraph
  int32_t node_list_id_ = -1;
  // Global SbpSignature List Size
  int32_t global_sbp_sig_size_ = -1;
  // Decide to use SbpSignature with this id
  int32_t final_sbp_sig_id_;
  // Available SbpSignature object for this node
  std::vector<NdSbpSignature> sbp_sig_list_;
  // Cost[sbp] is Computation Cost when using sbp_sig_list_[sbp]
  std::vector<double> cost_;
  std::vector<double> origin_cost_;

  // Child node list
  std::vector<SbpNode*> children_;
  // SbpSignature for each child node when using specific SbpSignature for this
  // node Its dimension is Number of Child Nodes * Number of Available
  // SbpSignatures for this node
  std::vector<std::vector<int32_t>> child_node_sbp_sig_;

  // Merge two nodes into this compound node
  std::vector<SbpNode*> half_node_;
  // We should delete those merged-signatures which has very large cost for speed up
  // New sbp_sig_list_ index map to each half_node_'s sig_index
  std::vector<std::pair<int32_t, int32_t>> merged_sig_id2half_sig_id_;

  std::vector<BinarySet> parallel_candidates_;

  OpNode* op_node_ = nullptr;

  // We divide the sbp graph into multiple layers.
  // min_layer_ is the minimum layer number to run this op as soon as possible.
  // max_layer_ is the maximum layer number without slowing down the whole process of the graph.
  // producer.max_layer_ < this_node.min_layer_ <= this_node.max_layer_ < consumer.min_layer_
  int32_t min_layer_ = -1, max_layer_ = -1;
  // Maximum layer in tributaries
  int32_t tributary_layer_ = -1;
  // Whether we are on the trunk
  bool on_trunk_ = false;
  // A counter_ buffer for topological traversal or something else
  int32_t counter_ = 0;
  // Accumulate trunk cost from consumer to the end
  double acc_trunk_cost_ = -1.0;

  // The produced blob belongs to the support of the total memory
  bool in_memory_support_ = false;
  // The consumed memory for different sbp strategies
  std::vector<int64_t> memory_;
  std::vector<int64_t> origin_memory_;
  // The weighted sum of time cost and memory cost
  // More specifically, weighted cost = time cost + kMemoryRatio * memory;
  // We do not add any weight for the time cost since we need to judge if a cost is less than
  // GetValidMaxCopyCost().
  std::vector<double> weighted_cost_;
  // Relationship between a merged node and its components
  HashMap<SbpNode*, std::vector<int32_t>> component2merged_sig_id2component_sig_id_;

  // Let one node point to another
  void StartPointToEnd(SbpNode* start_node, SbpNode* end_node);

  // Evaluate summery of cost in 1-ring neighborhood.
  double EvalNbhCost() const;
  // Drop down the maximum layer with the minimum layer from consumer
  void DropMaxLayer(int32_t upper_bound);
  // Drop down the available wait time with the minimum cost from downstream
  void DropAvailWaitTime(double curr_trunk_cost);
};  // class SbpNode

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_NODE_H_
