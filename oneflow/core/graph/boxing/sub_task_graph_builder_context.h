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
  TaskNode* GetProxyNode(TaskNode* src_node, const MemoryCase& src_mem_case, int64_t dst_machine_id,
                         const MemoryCase& dst_mem_case);
  template<typename T1, typename T2>
  void NaiveConnectAll121(const std::vector<T1*>& src_nodes, const std::vector<T2*>& dst_nodes) {
    CHECK_EQ(src_nodes.size(), dst_nodes.size());
    FOR_RANGE(int64_t, i, 0, dst_nodes.size()) {
      Connect<TaskNode>(src_nodes.at(i), task_graph()->NewEdge(), dst_nodes.at(i));
    }
  }

 private:
  TaskGraph* task_graph_;
  HashMap<TaskNode*, HashMap<std::pair<int64_t, MemoryCase>, TaskNode*>> node2proxies_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_CONTEXT_H_
