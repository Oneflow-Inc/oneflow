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

#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_CONSTRUCTOR_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_CONSTRUCTOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

class OpGraph;
class Job;

namespace auto_parallel {

// A constructor which will assemble the sbp_graph with the information from oneflow.
// SbpGraph contains the algorithms for elimination and search which is mainly for the strategy
// itself. Constructor mainly deal with the assemblage of each node, edge and the cost computation,
// activation of functions.
class SbpConstructor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SbpConstructor);
  SbpConstructor() = delete;
  SbpConstructor(const OpGraph& op_graph, Job* job)
      : cost_ratio_(job->job_conf().auto_parallel_computation_cost_ratio()),
        enable_trunk_algo_(job->job_conf().enable_auto_parallel_trunk_algo()),
        use_sbp_collector_(!Singleton<ResourceDesc, ForSession>::Get()
                                ->resource()
                                .disable_group_boxing_by_dst_parallel()
                           && job->job_conf().enable_auto_parallel_sbp_collector()),
        op_graph_(&op_graph) {
    sbp_graph_.SetWaitTime(job->job_conf().auto_parallel_wait_time());
    CHECK_JUST(Init(op_graph, job));
  }
  ~SbpConstructor() = default;

  Maybe<void> Init(const OpGraph& op_graph, Job* job);
  Maybe<void> FindBestSbpSignature();
  Maybe<void> DumpNdSbpSignatureForJob(const OpGraph& op_graph, Job* job);
  // Re-build OpGraph and check all sbp is same between op_graph and job
  Maybe<void> CheckSbpAgreement(const Job& job);
  // Print the graph with SBP in order
  void PrintSBPGraphDebugInfo();

 private:
  Maybe<void> InitSbpGraph(const OpGraph& op_graph, const Job& job);
  Maybe<void> GenerateNodeAndEdge(const OpGraph& op_graph, const Job& job);
  Maybe<void> FillSbpSignatureForOpNode(const OpGraph& op_graph, const Job& job);
  Maybe<void> StealSbpSignatureFromOpNode(const OpGraph& op_graph, const Job& job);
  Maybe<void> InitComputationCost(const OpGraph& op_graph);
  Maybe<void> InitCopyAndMemoryCost(const OpGraph& op_graph);
  Maybe<void> ApplyTrunkAlgo();
  Maybe<HashMap<const OpNode*, HashSet<std::string>>> GetMutableOpCtrlDeps(const OpGraph& op_graph);
  void InitAvailableMemory();
  void InitWeightedCost();
  // Load logical blob ids onto sbp edges
  void LoadLbi2SbpEdge(const OpGraph& op_graph);

  double cost_ratio_;
  bool enable_trunk_algo_;
  bool use_sbp_collector_;
  SbpGraph sbp_graph_;
  const OpGraph* op_graph_;
  HashMap<std::string, SbpNode*> op_name2sbp_node_;
  bool nccl_use_compute_stream_;
  int64_t available_memory_;
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_CONSTRUCTOR_H_
