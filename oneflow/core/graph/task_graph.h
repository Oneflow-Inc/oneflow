#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class TaskGraph final : public Graph<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskGraph);
  TaskGraph() = delete;
  ~TaskGraph() = default;

  TaskGraph(std::unique_ptr<const ChainGraph>&& chain_gph);

  void BldSubTskGphByBoxing(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);
  void BldSubTskGphByOneToOne(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);
  void BldSubTskGphBySelectOneSourceToSoleSink(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);

 private:
  TaskNode* AddCopyH2DTaskIfNotCpu(CompTaskNode*);
  TaskNode* AddCopyD2HTaskIfNotCpu(CompTaskNode*);
  void AddCopyCommNetTask(TaskNode* src, TaskNode* dst);
  void BuildOutBoxingIfNeed(
      const ChainNode*, const std::vector<CompTaskNode*>& sorted_comp_tasks,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_out_box);
  void BuildInBoxingIfNeed(
      const ChainNode*, const std::vector<CompTaskNode*>& sorted_comp_tasks,
      HashMap<const ChainNode*, std::vector<TaskNode*>>* chain2sorted_in_box);

  std::unique_ptr<const ChainGraph> chain_gph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
