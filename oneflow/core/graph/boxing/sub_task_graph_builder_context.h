#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

class SubTskGphBuilderCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilderCtx);
  explicit SubTskGphBuilderCtx(TaskGraph* task_graph);
  virtual ~SubTskGphBuilderCtx() = default;

  virtual TaskGraph* task_graph();
  TaskNode* CopyToMachine(TaskNode* src, const MemoryCase& src_mem_case, int64_t dst_machine_id);
  TaskNode* CopyToMachine(TaskNode* src, int64_t dst_machine_id);
  TaskNode* GetProxyNode(TaskNode* src_node, const MemoryCase& src_mem_case, int64_t dst_machine_id,
                         const MemoryCase& dst_mem_case);

 private:
  TaskGraph* task_graph_;
  HashMap<TaskNode*, HashMap<std::pair<int64_t, MemoryCase>, TaskNode*>> node2proxies_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
