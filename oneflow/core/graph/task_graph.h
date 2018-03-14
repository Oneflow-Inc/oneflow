#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;

  TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph);

  const char* TypeName() const override { return "TaskGraph"; }

#define DECLARE_BLD_SUB_TASK_GRAPH_METHOD(method_name)                         \
  void method_name(                                                            \
      const ChainNode* src_chain, const ChainNode* dst_chain,                  \
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,                 \
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,                 \
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,  \
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box, \
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);

  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByBoxing);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphByOneToOne);
  DECLARE_BLD_SUB_TASK_GRAPH_METHOD(BldSubTskGphBySelectOneSourceToSoleSink);

 private:
  TaskNode* AddCopyH2DTaskIfNotCpu(CompTaskNode*);
  TaskNode* AddCopyD2HTaskIfNotCpu(CompTaskNode*);
  void AddCopyCommNetTask(TaskNode* src, TaskNode* dst);
  void BuildOutBoxing(
      const ChainNode* chain,
      const std::vector<CompTaskNode*>& sorted_comp_tasks,
      std::vector<TaskNode*>* sorted_box,
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void BuildInBoxing(const ChainNode* chain,
                     const std::vector<CompTaskNode*>& sorted_comp_tasks,
                     std::vector<TaskNode*>* sorted_box,
                     std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);

  std::unique_ptr<const ChainGraph> chain_gph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
