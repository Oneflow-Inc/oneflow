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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_STATISTICS_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_STATISTICS_H_

#include "oneflow/core/auto_parallel/sbp_graph.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {
namespace auto_parallel {

class SbpStatistics final {
 public:
  ~SbpStatistics() = default;

  explicit SbpStatistics(double transfer_cost, double wait_time);

  // Collect statistics
  void CollectStatistics(const SbpGraph<cfg::NdSbpSignature>& sbp_graph);
  void CollectStatistics(const SbpNode<cfg::NdSbpSignature>& sbp_node);
  void CollectStatistics(const SbpEdge<cfg::NdSbpSignature>& sbp_edge);

  // Print out the statistics information
  void PrintStatistics();

 private:
  // Parameters for sbp graph
  double transfer_cost_;
  double wait_time_;
  // Total number of operators
  int32_t op_num_ = 0;
  // Total cost for varification, which should be the same as sbp_graph.ComputeCost()
  double total_cost_ = 0.0;
  // Communication cost without fixed transfer_cost, but still have wait time
  double real_copy_cost_ = 0.0;
  // Total cost for communication
  double total_copy_cost_ = 0.0;
  // Total cost for computation
  double total_comp_cost_ = 0.0;
  // Number of edge with communication
  int32_t num_comm_ = 0;
  // Number of edge with slight communication
  int32_t num_slight_comm_ = 0;
};

}  // namespace auto_parallel

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_STATISTICS_H_