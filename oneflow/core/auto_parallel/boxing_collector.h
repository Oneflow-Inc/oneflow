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

#ifndef BOXING_COLLECTOR_
#define BOXING_COLLECTOR_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "sbp_util.h"

namespace oneflow {

class BoxingCollector {
 public:
  explicit BoxingCollector(const OpGraph& op_graph);

  ~BoxingCollector(){};

  // Collect all the possible Sbp Parallel from an OpGraph
  void CollectUniverse(const OpGraph& op_graph);
  // Generate the transfer rule for different combinations and hierarchies
  Maybe<void> GenerateCombination();
  // Print the cost and middle nodes
  void PrintBoxingTables();

 private:
  // Stores all the possible cfg::NdSbp.
  std::unordered_map<::oneflow::cfg::SbpParallel, int32_t> SbpParallelUniverse;
  // Relationship between id and Sbp Parallel
  std::vector<::oneflow::cfg::SbpParallel> id2SbpParallel;
  // minimum cost
  // minimum_copy_cost[producer][consumer]
  std::vector<std::vector<double>> minimum_copy_cost;
  // middle nodes
  // middle_nodes[producer][consumer][different choices] is a vector of middle nodes
  // middle_nodes[producer][consumer][different choices].size() is the minimum number of middle
  // nodes that needs to be inserted
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> middle_nodes;
  // Relationship between id and Nd Sbp
  std::vector<cfg::NdSbp> nd_sbp_lists;
};  // class BoxingCollector

}  // namespace oneflow

#endif  // BOXING_COLLECTOR_