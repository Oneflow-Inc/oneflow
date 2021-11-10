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

class SbpConstructor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SbpConstructor);
  SbpConstructor() = delete;
  SbpConstructor(const OpGraph& op_graph, Job* job)
      : cost_ratio_(job->job_conf().auto_parallel_computation_cost_ratio()),
        wait_time_(job->job_conf().auto_parallel_wait_time()),
        transfer_cost_(job->job_conf().auto_parallel_transfer_cost()),
        use_sbp_collector_(!Global<ResourceDesc, ForSession>::Get()
                                ->resource()
                                .disable_group_boxing_by_dst_parallel()) {
    CHECK_JUST(Init(op_graph, job));
  }
  ~SbpConstructor() = default;

  Maybe<void> Init(const OpGraph& op_graph, Job* job);
  Maybe<void> FindBestSbpSignature();
  Maybe<void> UpdateSbpSignatureForJob(const OpGraph& op_graph);
  // Print the graph with SBP in order
  void PrintSBPGraphDebugInfo();

 private:
  Maybe<void> InitSbpGraph(const OpGraph& op_graph, const Job& job);
  Maybe<void> GenerateNodeAndEdge(const OpGraph& op_graph);
  Maybe<void> FillSbpSignatureForOpNode(const OpGraph& op_graph, const Job& job);
  Maybe<void> InitComputationCost(const OpGraph& op_graph);
  Maybe<void> InitCopyCost(const OpGraph& op_graph);
  // Load logical blob ids onto sbp edges
  void LoadLbi2SbpEdge(const OpGraph& op_graph);

  double cost_ratio_;
  double wait_time_;
  double transfer_cost_;
  bool use_sbp_collector_;
  SbpGraph<cfg::SbpSignature> sbp_graph_;
  HashMap<std::string, SbpNode<cfg::SbpSignature>*> op_name2sbp_node_;
};

}  // namespace auto_parallel
}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_CONSTRUCTOR_H_
