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

namespace oneflow {

// deciding parameter
// The sorting order of nodes for the straighten algorithm
enum StraightenOrder : int {
  kTributaryLayerAscend = 0,     // small tributary layers go first
  kDistanceToOverlapAscend = 1,  // small minimum distance to overlap go first
  kLayerAscend = 2,              // first in first out
  kMemoryIncrementAscend = 3,    // small memory increment go first
  kExceedTimeAscend = 4,         // small exceed time go first

  kTributaryLayerDescend = 100,     // large tributary layers go first
  kDistanceToOverlapDescend = 101,  // long distance to overlap go first
  kLayerDescend = 102,              // last in first out
  kMemoryIncrementDescend = 103,    // large memory increment go first
  kExceedTimeDescend = 104,         // large exceed time go first
};

// The difference between a descending order and its corresponding ascending order
const int kDiff4AscendDescend = 100;

// Some operators have longer time in cpu and less time in gpu.
// Running those operators without overlap would cause large gap during each iteration.
// For example, expand dims would not execute any kernel on gpu but still need 10us to execute some
// functions on cpu.
bool ShortGpuTime(const OperatorConf& op_conf);

// SAT, a.k.a. Scholastic Aptitude Test,
// is the college admission test in the United States of America.
void InitDecideParameters(StraightenAlgorithmTag sat,
                          std::vector<StraightenOrder>* decide_parameters);

template<class HashMapType>
void UpdateSat(const HashMapType& task_node2topo_struct, StraightenAlgorithmTag* sat) {
  *sat = GlobalJobDesc().job_conf().straighten_algorithm_tag_in_task_graph();
  if (*sat == StraightenAlgorithmTag::kOverlap4CpuGpu) {
    // If not cpu nodes, then the overlap strategy between cpu and gpu might consume large memory
    bool exist_cpu_nodes = false;
    for (const auto& pair : task_node2topo_struct) {
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

void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STRAIGHTEN_NODES_H_
