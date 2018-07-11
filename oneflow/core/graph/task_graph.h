#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/logical_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;

  TaskGraph(std::unique_ptr<const LogicalGraph>&& logical_gph);

  const char* TypeName() const override { return "TaskGraph"; }
  void AddOrderingCtrlEdgeInSameChain();
  void AddMutexCtrlEdgeInSameChain();
  void AddOrderCtrlEdgeBetweenCopyAndMdUpdt();
  void AcyclicTopoForEachNode(std::function<void(TaskNode* node)> handler) const;

  using MutBufTaskFn = std::function<TaskNode**(CompTaskNode*, int64_t, int32_t)>;

 private:
  void BldSubTskGphByBoxing(
      const LogicalNode* src_logical, const LogicalNode* dst_logical,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
      HashMap<const LogicalNode*, std::vector<TaskNode*>>* logical2sorted_in_box,
      HashMap<const LogicalNode*, std::vector<TaskNode*>>* logical2sorted_out_box,
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly);

  void BldSubTskGphByOneToOne(const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                              const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                              MutBufTaskFn MutBufTask);

  void BldSubTskGphBySelectOneSourceToSoleSink(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask,
      std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly);

  void BldSubTskGphByReduceScatter2ReduceLocalAdd(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask);

  void BldSubTskGphByReduceScatter2ReduceGlobalAdd(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask);

  void BldSubTskGphByReduceLocalAdd2ReduceGlobalAdd(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask);

  void BldSubTskGphByReduceGlobalAdd2ReduceGather(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, MutBufTaskFn MutBufTask);

  void BuildTaskPath(CompTaskNode* src, CompTaskNode* dst, MutBufTaskFn MutBufTask,
                     bool use_buf_task_node);
  TaskNode* BuildTaskStep(TaskNode* cur_node, TaskNode* dst,
                          std::function<TaskNode*(int64_t, int32_t)> GetBufTask,
                          std::function<TaskNode*(int64_t, int32_t, TaskNode*)> SetBufTask,
                          bool use_buf_task_node);
  TaskNode* AddCopyH2DTaskTo(TaskNode*);
  TaskNode* AddCopyD2HTaskFrom(TaskNode*);
  TaskNode* AddCopyCommNetTaskBetween(TaskNode* src, TaskNode* dst);
  void BuildOutBoxing(const LogicalNode* logical,
                      const std::vector<CompTaskNode*>& sorted_comp_tasks,
                      std::vector<TaskNode*>* sorted_out_box,
                      std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void BuildInBoxing(const LogicalNode* logical,
                     const std::vector<CompTaskNode*>& sorted_comp_tasks,
                     std::vector<TaskNode*>* sorted_in_box,
                     std::function<int64_t(const TaskNode*)> AllocateCpuThrdId);
  void ConnectWithCopyCommNetIfNeed(TaskNode* src, TaskNode* dst);

  void SetAreaIdForNewNodes(const LogicalNode* src_logical, const LogicalNode* dst_logical);
  void CollectAncestorsForEachNode();
  void FindChainsInSameStream();

  std::unique_ptr<const LogicalGraph> logical_gph_;
  std::vector<TaskNode*> ordered_task_nodes_;
};
bool IsBackEdge(TaskNode* src, TaskNode* dst);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
