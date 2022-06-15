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
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

class OpGraph;
class Job;

enum TaskClassifier : int {
  kWaitingTransfer = 0,
  kWaitingComputation = 1,
  kRunASAP = 2,
  kRunALAP = 3
};

bool IsTransferNode(TaskType task_type);

// Classifier for the set according to the task type
TaskClassifier GetTaskClassifier(const TaskNode* node);

class TopoStruct {
 public:
  TaskNode* node;
  int32_t min_layer = -1;
  int32_t tributary_layer = -1;
  bool on_mainstem = false;
  int32_t counter = 0;
  int32_t min_distance2transfer = -1;
  TopoStruct* next_same_node = nullptr;
  // We can have some other nodes in it for example
  // SbpNode<NdSbpSignature>* node;
  // SbpEdge<NdSbpSignature>* node;
  // Or we can omit all the pointers and leave all the useful parameters.

  // Drop down the tributary layer
  void DropTributaryLayer(int32_t upper_bound);

  void SpreadTributaryLayer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  void SpreadMainstem(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // The minimum computation distance from the beginning of this op to the next transfer
  int32_t GetMinDistance2Transfer(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

  // deciding parameter
  // i = 0: those with small tributary layers go first
  // i = 1: those with small minimum distance to transfer go first
  // i = 2: first in first out
  // i = 3: those with large tributary layers go first
  // i = 4: those with long distance to transfer go first
  // i = 5: last in first out
  int32_t GetDecidingParameter(int32_t i) const;
};

void FindMainstem(HashMap<TaskNode*, TopoStruct>* task_node2topo_struct);

void StraightenNodes(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STRAIGHTEN_NODES_H_
