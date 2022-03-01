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
#include "sbp_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_sig_infer_hint.h"
#include "oneflow/core/job/job_builder.h"
// #include "sbp_constructor.h"
#define DEBUG_COLLECTOR_

namespace oneflow {

namespace auto_parallel {

class SbpCollector {
 public:
  // Stores all the possible NdSbp.
  std::unordered_map<::oneflow::NdSbp, int32_t> SbpParallelUniverse;
  // Relationship between id and Sbp Parallel
  std::vector<::oneflow::NdSbp> id2SbpParallel;
  // Calculate number of downstream sbp
  std::vector<int32_t> accumulator;
  // A binary set buffer to indicate sets of downstream sbp
  BinarySet bs_buffer;

  SbpCollector();

  ~SbpCollector() {}

  // Collect all the possible Sbp Parallel from a NdSbpSignature
  void CollectUniverse(NdSbpSignature& sbp_);
  // Collect all the possible Sbp Parallel from a SbpNode
  void CollectUniverse(SbpNode<NdSbpSignature>* sbp_node);
  // Collect all the possible Sbp Parallel from a SbpGraph
  void CollectUniverse(SbpGraph<NdSbpSignature>& sbp_graph);
  // Initialize sbp proxy with given parallel candidates of a blob
  SbpNode<NdSbpSignature>* InitializePorxy(
      SbpGraph<NdSbpSignature>& sbp_graph,
      std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates);

  // Initialize copy cost from producer to proxy of producer
  void InitializeCopyCostFromNode2Proxy(SbpNode<NdSbpSignature>* sbp_proxy,
                                        const LogicalBlobId& lbi);

  // Initialize copy cost from proxy of producer to consumers
  void InitializeCopyCostFromProxy2Consumer(
      SbpNode<NdSbpSignature>* sbp_proxy,
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set,
      HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node);

  // Export list of possible combination of Sbp Parallels
  void ProxySbpCandidate(const OpGraph& op_graph,
                         HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node,
                         SbpGraph<NdSbpSignature>& sbp_graph);

 private:
  // Depth first search to collect Sbp Parallel information for different lbis
  void DfsSbpSet(
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>::iterator it,
      HashMap<std::pair<std::string, std::string>, std::unordered_set<int32_t>>&
          consumer_bn2sbp_set,
      HashMap<std::string, SbpNode<NdSbpSignature>*>& op_name2sbp_node,
      std::unordered_set<BinarySet, BinarySetHasher>& ParallelCandidates);
};  // class SbpCollector

}  // namespace auto_parallel

}  // namespace oneflow

#endif  // SBP_COLLECTOR_
