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

#ifndef ONEFLOW_CORE_AUTO_PARALLEL_BOXING_COLLECTOR_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_BOXING_COLLECTOR_H_

#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/sbp_infer_util.h"

namespace oneflow {

class BoxingCollector final {
 public:
  BoxingCollector() = default;

  ~BoxingCollector() = default;

  // A constructor with init, designed for uncustomized boxing collector
  BoxingCollector(int32_t max_axis);

  // Set default Sbp list
  void CollectUniverse(int32_t max_axis);

  // Construct a boxing collector with given maximum number of axis
  Maybe<void> Init(int32_t max_axis);

  // Generate nd sbp list
  void GenerateNdSbpList();
  // Generate the transfer rule for different combinations and hierarchies
  Maybe<void> GenerateCombination(int32_t max_middle_node_num);
  // Print the cost and middle nodes
  void PrintBoxingTables();
  // Ask if the boxing algorithm accepts the current sbp combination
  // If is_customized is true and we can not find a middle node list with
  // resonable cost, error occurs.
  // If compute_cost is true, then no error occur even if no suitable middle nodes paths found.
  Maybe<void> AskSbpCombination(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                const BlobDesc& logical_blob_desc,
                                const ParallelDesc& producer_parallel_desc,
                                const ParallelDesc& consumer_parallel_desc, bool is_customized,
                                std::vector<NdSbp>& middle_sbps, bool compute_cost);
  // Filter nd sbp from nd_sbp_lists_ with given logical shape
  Maybe<void> FilterNdSbpList4LogicalShape(const BlobDesc& logical_blob_desc,
                                           const Shape& parallel_hierarchy);

 private:
  // Collect Sbp Parallel
  void CollectUniverse(const SbpParallel& sbp);
  // Stores all the possible SbpParallel.
  HashMap<::oneflow::SbpParallel, int32_t> SbpParallelUniverse_;
  // Relationship between id and Sbp Parallel
  std::vector<::oneflow::SbpParallel> id2SbpParallel_;
  // minimum cost
  // minimum_copy_cost[producer][consumer]
  std::vector<std::vector<double>> minimum_copy_cost_;
  // middle nodes
  // middle_nodes_[producer][consumer][different choices] is a vector of middle nodes
  // middle_nodes_[producer][consumer][different choices].size() is the minimum number of middle
  // nodes that needs to be inserted
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> middle_nodes_;
  // Stores all the possible NdSbp.
  std::unordered_map<::oneflow::NdSbp, int32_t> NdSbpUniverse_;
  // Relationship between id and Nd Sbp
  std::vector<NdSbp> nd_sbp_lists_;
};  // class BoxingCollector

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_BOXING_COLLECTOR_H_
