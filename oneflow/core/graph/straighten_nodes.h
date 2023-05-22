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
#ifndef ONEFLOW_CORE_GRAPH_STRAIGHTEN_NODES_H_
#define ONEFLOW_CORE_GRAPH_STRAIGHTEN_NODES_H_

#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/job/job_conf.pb.h"

namespace oneflow {

// The difference between a descending order and its corresponding ascending order
const int kDiff4AscendDescend = 100;

// deciding parameter
// The sorting order of nodes for the straighten algorithm
enum StraightenOrder : int {
  kTributaryLayerAscend = 0,     // small tributary layers go first
  kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  kLayerAscend = 2,              // first in first out
  kMemoryIncrementAscend = 3,    // small memory increment go first
  kExceedTimeAscend = 4,         // small exceed time go first
  kMemoryVolumeAscend = 5,       // small memory volume go first
  kMaxLayerAscend = 6,           // the urgent one go first

  kTributaryLayerDescend =
      kDiff4AscendDescend + kTributaryLayerAscend,  // large tributary layers go first
  kDistanceToOverlapDescend =
      kDiff4AscendDescend + kDistanceToOverlapAscend,  // long distance to overlap go first
  kLayerDescend = kDiff4AscendDescend + kLayerAscend,  // last in first out
  kMemoryIncrementDescend =
      kDiff4AscendDescend + kMemoryIncrementAscend,              // large memory increment go first
  kExceedTimeDescend = kDiff4AscendDescend + kExceedTimeAscend,  // large exceed time go first
  kMemoryVolumeDescend = kDiff4AscendDescend + kMemoryVolumeAscend,  // large memory volume go first
  kMaxLayerDescent = kDiff4AscendDescend + kMaxLayerAscend,          // the non-urgent one go first
};

// Some operators have longer time in cpu and less time in gpu.
// Running those operators without overlap would cause large gap during each iteration.
// For example, expand dims would not execute any kernel on gpu but still need 10us to execute some
// functions on cpu.
bool ShortGpuTime(const OperatorConf& op_conf);

// SAT, a.k.a. Scholastic Aptitude Test,
// is the college admission test in the United States of America.
void InitDecideParameters(StraightenAlgorithmTag sat,
                          std::vector<StraightenOrder>* decide_parameters);

// Maximum overlap number
// While running an overlap operator, we would run some other operators simultaneously.
int32_t MaximumOverlapNum(StraightenAlgorithmTag sat, bool nccl_use_compute_stream);

template<class HashMapType>
void UpdateSat(const HashMapType& node2topo_struct, StraightenAlgorithmTag* sat) {
  *sat = GlobalJobDesc().job_conf().straighten_algorithm_tag_in_task_graph();
  if (*sat == StraightenAlgorithmTag::kOverlap4CpuGpu) {
    // If not cpu nodes, then the overlap strategy between cpu and gpu might consume large memory
    bool exist_cpu_nodes = false;
    for (const auto& pair : node2topo_struct) {
      // Found a cpu node
      if (pair.second.exceed_time == 1) {
        exist_cpu_nodes = true;
        break;
      }
    }
    if (!exist_cpu_nodes) {
      // Switch to the compress memory strategy, the default one
      // Since the overlap strategy for transfer might not be working on 1n1d.
      *sat = StraightenAlgorithmTag::kCompressMemory;
    }
  }
}

// Make sure that we use the same boolean value nccl_use_compute_stream through the straighten
// algorithm
void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes,
                     bool nccl_use_compute_stream);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STRAIGHTEN_NODES_H_
