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

  // A constructor with init, designed for non-customized boxing collector
  BoxingCollector(int32_t max_axis);

  // Set default Sbp list
  void CollectUniverse(int32_t max_axis);

  // Construct a boxing collector with given maximum number of axis
  Maybe<void> Init(int32_t max_axis);
  // Init with given blob description
  Maybe<void> Init(const BlobDesc& logical_blob_desc, const ParallelDesc& parallel_desc);

  // Generate nd sbp list
  void GenerateNdSbpList(int32_t hierarchy_num);
  // Generate the map from 1d sbp to 2d sbp
  void GenerateMap1d2nd();
  // Generate the transfer rule for different combinations with the same hierarchy
  Maybe<void> GenerateCombination4SamePlacement(int32_t max_middle_node_num);
  Maybe<void> GenerateCombination4SamePlacement(int32_t max_middle_node_num,
                                                const BlobDesc& blob_desc,
                                                const ParallelDesc& parallel_desc);
  // Generate the transfer rule for different combinations with different hierarchies
  // on the same placement
  Maybe<void> GenerateCombination4DiffHierarchy(BoxingCollector* boxing_collector_producer,
                                                BoxingCollector* boxing_collector_consumer);
  // Generate the transfer rule for different combinations with different placements
  Maybe<void> GenerateCombination4DiffPlacement(BoxingCollector* boxing_collector_producer,
                                                BoxingCollector* boxing_collector_consumer);
  Maybe<void> GenerateCombination4DiffPlacement(BoxingCollector* boxing_collector_producer,
                                                BoxingCollector* boxing_collector_consumer,
                                                const BlobDesc& blob_desc,
                                                const ParallelDesc& in_parallel_desc,
                                                const ParallelDesc& out_parallel_desc);
  // Print the cost and middle nodes
  void PrintBoxingTables();
  // Ask if the boxing algorithm accepts the current sbp combination
  // If is_customized is true and we can not find a middle node list with
  // reasonable cost, error occurs.
  // If compute_cost is true, then no error occur even if no suitable middle nodes paths found.
  // For different placements, we would return a diagonal node.
  // Before this diagonal node (< *diag_node_pos), we use the parallel description of the producer.
  // After this diagonal node (>= *diag_node_pos), we use the parallel description of the consumer.
  Maybe<void> AskSbpCombination(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                const BlobDesc& logical_blob_desc,
                                const ParallelDesc& producer_parallel_desc,
                                const ParallelDesc& consumer_parallel_desc, bool is_customized,
                                std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos,
                                bool compute_cost);
  // Filter nd sbp from nd_sbp_lists_ with given logical shape
  Maybe<void> FilterNdSbpList4LogicalShape(const BlobDesc& logical_blob_desc,
                                           const Shape& parallel_hierarchy);

 private:
  // Collect Sbp Parallel
  void CollectUniverse(const SbpParallel& sbp);
  // Find corresponding id for Nd sbp
  int32_t FindId4NdSbp(const NdSbp& nd_sbp);
  // Ask for sbp combination with the same 2-D hierarchy and placement
  Maybe<void> AskSbpCombination4Same2DPlacement(const NdSbp& sbp_producer,
                                                const NdSbp& sbp_consumer,
                                                const BlobDesc& logical_blob_desc,
                                                const ParallelDesc& producer_parallel_desc,
                                                const ParallelDesc& consumer_parallel_desc,
                                                bool is_customized, std::vector<NdSbp>& middle_sbps,
                                                int32_t* diag_node_pos, bool compute_cost);
  // Ask for sbp combination with different hierarchies on the same placement
  Maybe<void> AskSbpCombination4DiffPlacement(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                              const BlobDesc& logical_blob_desc,
                                              const ParallelDesc& producer_parallel_desc,
                                              const ParallelDesc& consumer_parallel_desc,
                                              bool is_customized, std::vector<NdSbp>& middle_sbps,
                                              int32_t* diag_node_pos, bool compute_cost);
  // Generate the transfer rule for one combination with different hierarchies on the same
  // placement. id_producer -> id_consumer.
  Maybe<void> Generate1Combination4DiffHierarchy(int32_t id_producer, int32_t id_consumer,
                                                 BoxingCollector* boxing_collector_producer,
                                                 BoxingCollector* boxing_collector_consumer,
                                                 std::vector<std::vector<int32_t>>& diag_nodes);
  // The cost for transferring a 1D sbp between different placements
  Maybe<void> ComputeCostFor1DSbpDiffPlacement(
      const BlobDesc& blob_desc, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc,
      std::vector<std::vector<double>>& cost_4_diff_placement);
  // Generate the transfer rule for one combination with different placements
  // id_producer -> id_consumer.
  Maybe<void> Generate1Combination4DiffPlacement(
      int32_t id_producer, int32_t id_consumer, BoxingCollector* boxing_collector_producer,
      BoxingCollector* boxing_collector_consumer,
      const std::vector<std::vector<double>>& cost_4_diff_placement,
      std::vector<std::vector<int32_t>>& diag_nodes);
  // Ask for one combination with different hierarchies and placements
  Maybe<bool> Ask1Combination4DiffPlacement(const NdSbp& sbp_producer, const NdSbp& sbp_consumer,
                                            const BlobDesc& logical_blob_desc,
                                            const ParallelDesc& producer_parallel_desc,
                                            const ParallelDesc& consumer_parallel_desc,
                                            bool is_customized, std::vector<NdSbp>& middle_sbps,
                                            int32_t* diag_node_pos, bool compute_cost,
                                            BoxingCollector* boxing_collector_producer,
                                            BoxingCollector* boxing_collector_consumer,
                                            const std::vector<std::vector<int32_t>>& diag_nodes);
  // Ask for sbp combination for general basic communication
  Maybe<void> AskSbpCombination4GeneralBasicCommunication(
      const NdSbp& sbp_producer, const NdSbp& sbp_consumer, const BlobDesc& logical_blob_desc,
      const ParallelDesc& producer_parallel_desc, const ParallelDesc& consumer_parallel_desc,
      std::vector<NdSbp>& middle_sbps, int32_t* diag_node_pos);
  // Ask for a all-split sbp which is closed to the original one
  Maybe<void> AskCloseAllSplitSbp(const NdSbp& nd_sbp, const ParallelDesc& parallel_desc,
                                  const BlobDesc& logical_blob_desc,
                                  std::vector<NdSbp>& middle_sbps);
  // Stores all the possible SbpParallel.
  HashMap<SbpParallel, int32_t> sbp_parallel_universe_;
  // Relationship between id and Sbp Parallel
  std::vector<SbpParallel> id2sbp_parallel_;
  // minimum cost
  // minimum_copy_cost[producer][consumer]
  std::vector<std::vector<double>> minimum_copy_cost_;
  // middle nodes
  // middle_nodes_[producer][consumer][different choices] is a vector of middle nodes
  // middle_nodes_[producer][consumer][different choices].size() is the minimum number of middle
  // nodes that needs to be inserted
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> middle_nodes_;
  // Stores all the possible NdSbp.
  std::unordered_map<NdSbp, int32_t> nd_sbp_universe_;
  // Relationship between id and Nd Sbp
  std::vector<NdSbp> nd_sbp_lists_;
  // The diagonal middle node for different placements
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> diag_node_diff_placement_;
  // The diagonal middle node for different hierarchies in the same placement
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> diag_node_diff_hierarchy_;
  // Id Map from 1d sbp to 2d sbp
  // For example: B -> (B, B), S0 -> (S0, S0)
  std::vector<int32_t> id_1d_2_nd_;
  // The sbp size in the combination table
  int32_t hierarchy_num_;
  // How the boxing collector is initialized
  int32_t init_type_ = -1;
  // Enable general basic communication or not
  const bool enable_general_basic_communication =
      ParseBooleanFromEnv("ONEFLOW_BOXING_ENABLE_GENERAL_BASIC_COMMUNICATION", false);
};  // class BoxingCollector

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_BOXING_COLLECTOR_H_
