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
#ifndef ONEFLOW_CORE_GRAPH_STRAIGHTEN_MEMORY_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_STRAIGHTEN_MEMORY_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

// Straighten the task graph to reduce memory
void StraightenMemoryTaskGraph(TaskGraph* task_graph, std::vector<TaskNode*>* ordered_task_nodes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STRAIGHTEN_MEMORY_TASK_GRAPH_H_