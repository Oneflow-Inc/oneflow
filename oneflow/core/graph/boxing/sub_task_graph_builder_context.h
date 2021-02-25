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
#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_

#include "oneflow/core/common/men_zone_id_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

class TaskGraph;
class TaskNode;

class SubTskGphBuilderCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilderCtx);
  explicit SubTskGphBuilderCtx(TaskGraph* task_graph);
  virtual ~SubTskGphBuilderCtx() = default;

  virtual TaskGraph* task_graph();
  TaskNode* GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id, int64_t dst_machine_id,
                         MemZoneId dst_mem_zone_id);
  TaskNode* GetProxyNode(TaskNode* src_node, MemZoneId src_mem_zone_id,
                         const ParallelDesc& dst_parallel_desc, const int64_t dst_parallel_id);
  template<typename T1, typename T2>
  void ConnectAll121(const std::vector<T1*>& src_nodes, const std::vector<T2*>& dst_nodes) {
    CHECK_EQ(src_nodes.size(), dst_nodes.size());
    FOR_RANGE(int64_t, i, 0, dst_nodes.size()) {
      Connect<TaskNode>(src_nodes.at(i), task_graph()->NewEdge(), dst_nodes.at(i));
    }
  }

 private:
  TaskGraph* task_graph_;
  HashMap<TaskNode*, HashMap<std::pair<int64_t, int64_t>, TaskNode*>> node2proxies_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
