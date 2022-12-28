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

#ifndef SBP_COLLECTOR_
#define SBP_COLLECTOR_

#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/local_sig_infer_hint.h"
#include "oneflow/core/job/job_builder.h"
// #include "sbp_constructor.h"
#define DEBUG_COLLECTOR_

namespace oneflow {

namespace auto_parallel {

class SbpCollector {
 public:
  SbpCollector();

  ~SbpCollector() {}

  // Collect all the possible Sbp Parallel from a SbpGraph
  void CollectUniverse(const SbpGraph& sbp_graph);

  // Export list of possible combination of Sbp Parallels
  void ProxySbpCandidate(const OpGraph& op_graph,
                         const HashMap<std::string, SbpNode*>& op_name2sbp_node,
                         SbpGraph& sbp_graph);

 private:
  // Stores all the possible NdSbp.
  std::unordered_map<NdSbp, int32_t> nd_sbp_universe_;
  // Relationship between id and Sbp Parallel
  std::vector<NdSbp> id2nd_sbp_;
  // Calculate number of downstream sbp
  std::vector<int32_t> accumulator_;
  // A binary set buffer to indicate sets of downstream sbp
  BinarySet bs_buffer_;

  // Collect all the possible Sbp Parallel from a NdSbpSignature
  void CollectUniverse(const NdSbpSignature& nd_sbp_sig);
  // Collect all the possible Sbp Parallel from a SbpNode
  void CollectUniverse(const SbpNode* sbp_node);

  // Initialize copy cost from producer to proxy of producer
  void InitializeCopyCostFromNode2Proxy(const SbpNode* sbp_proxy, const LogicalBlobId& lbi) const;

  // Initialize copy cost from proxy of producer to consumers
  void InitializeCopyCostFromProxy2Consumer(
      SbpNode* sbp_proxy,
      const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
      const HashMap<std::string, SbpNode*>& op_name2sbp_node) const;

  // Maximum number of possible sbp in the proxy
  const unsigned long max_num_sbp_proxy_ = 3;

  // Depth first search to collect Sbp Parallel information for the whole sbp set
  void DfsSbpSet(int32_t depth, int32_t max_depth, const std::unordered_set<int32_t>& sbp_sets,
                 const std::unordered_set<int32_t>::iterator& sbp_set_it,
                 const HashMap<std::pair<std::string, std::string>, BinarySet>& consumer_bn2sbp_set,
                 const std::vector<BinarySet>& unique_sbp_groups,
                 std::vector<BinarySet>& parallel_candidates);
};  // class SbpCollector

}  // namespace auto_parallel

}  // namespace oneflow

#endif  // SBP_COLLECTOR_
