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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_

#include <assert.h>
#include <algorithm>
#include <unordered_set>
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/auto_parallel/sbp_node.h"
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {
namespace auto_parallel {

class SbpEdge final {
  /* There are 3 types of edges:
   * 1. start_node_ -> end_node_
   *      Nothing special
   * 2. Multiple start_node_ -> end_node_
   *      edge_list_ will store all the edges which goes from start_node_ to end_node_
   * 3. start_node_ -> mid_node_ -> end_node_
   *      It will pass by a middle node.
   */
 public:
  // Constructor for type 1 & 2
  SbpEdge(SbpNode* start_node, SbpNode* end_node) : start_node_(start_node), end_node_(end_node) {
    mid_node_ = nullptr;
  }
  // Constructor for type 3
  SbpEdge(SbpNode* start_node, SbpNode* mid_node, SbpNode* end_node, SbpEdge* first_edge,
          SbpEdge* second_edge);

  // Deconstructor
  ~SbpEdge() {
    if (mid_node_ != nullptr) { delete mid_node_; }
    for (auto& this_edge : edge_list_) { delete this_edge; }
  }

  // Update copy cost for type 2 and 3
  void SummarizeCost();
  // Duplicate Cost. Designed for merging two nodes.
  void DuplicateCost(bool merged_node_is_start_node, bool duplicating_first_node,
                     const std::vector<std::pair<int32_t, int32_t>>& merged_sig_id2children_sig_id);
  // Determine Final SbpSignature for attachment of this edge
  void FinalizeSbp();
  // Use Greedy Strategy to pick the sbp signature with minimum cost for this
  // edge. You should have an initial strategy before running this. And the
  // graph should be fully eliminated.
  double GreedyStrategy();

  // load a logical blob
  void LoadLbi(const LogicalBlobId& lbi);

  // check the existence of a logical blob
  bool SearchLbi(const LogicalBlobId& lbi) const;

  // unload a logical blob
  void UnloadLbi(const LogicalBlobId& lbi);

  // Not carrying any blob
  bool EmptyLbi() const;

  // Get the minimum element in Cost
  double GetMinCost();
  // Get the maximum element in Cost
  double GetMaxCost() const;

  // Assemble copy cost
  // compute_cost = true: It is computing cost
  // compute_cost = false: It is deciding whether this edge needs the wait time.
  void InitializeCopyCost(const std::string& ibn, bool compute_cost, bool use_sbp_collector);

  // find the cut ratio
  // (#c>GetValidMaxCopyCost() in Cost)/(#c in Cost)
  // But we would lift the cut ratio to 1 to filter out some improper couples
  double FindCutRatio(int32_t threshold) const;
  // Get the cut ratio
  double GetCutRatio() const;

 private:
  friend class SbpNode;
  friend class SbpGraph;
  friend class SbpCollector;
  friend class SbpConstructor;

  // The edge point from start_node_ to end_node_
  // It will have a middle node if and only if type 3
  SbpNode *start_node_, *mid_node_, *end_node_;
  // Cost[sbp_i][sbp_j] is the total cost from start_node_ with sbp_i to end_node_
  // with sbp_j
  std::vector<std::vector<double>> cost_;
  // SbpSignature for mid_node_ with corresponding Cost if type 3, empty otherwise
  std::vector<std::vector<int32_t>> mid_node_sbp_sig_;
  // Contained edge list:
  // empty if type 1,
  // Parallel edges if type 2,
  // succeed edges if type 3
  // the edge list might have reverse direction:
  // example 1: type 3 edge_list_ contain two edges:
  //        mid_node_ -> start_node_, mid_node_ -> end_node_;
  // example 2: type 2 edge_list_ contain three edges:
  //        start_node_ -> end_node_, end_node_ -> start_node_, start_node_ -> end_node_;
  std::vector<SbpEdge*> edge_list_;
  // Time waiting for other gpus. pthread_cond_wait
  double wait_time_ = -1.0;

  // a set of ids of logical blobs carried/transferred on this sbp edge
  std::unordered_set<LogicalBlobId> carry_lbis_;

  // Minimum and maximum cost would not be changed by eliminations, which will generate new edges.
  // Also would not be changed by node merging, which will only perform cost copy for the expanding
  // dimensions.
  // Minimum cost in the 2D array Cost.
  // Would be initialized after GetMinCost();
  // Only used in the final graph.
  double min_cost_ = -1.0;
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_EDGE_H_
