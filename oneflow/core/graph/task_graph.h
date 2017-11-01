#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/copy_task_node.h"
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

  void BldSubTskGphByNormalBoxing(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks);
  void BldSubTskGphByAddCloneBoxing(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks);
  void BldSubTskGphByOneToOne(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks);
  void BldSubTskGphBySelectOneSourceToSoleSink(
      const ChainNode* src_chain, const ChainNode* dst_chain,
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks);

 private:
  void BldSubTskGphByBoxing(
      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
      std::function<void(BoxingOpConf*)> BoxingOpConfSetter);

  TaskNode* AddCopyH2DTaskIfNotCpu(CompTaskNode*);
  TaskNode* AddCopyD2HTaskIfNotCpu(CompTaskNode*);
  void AddCopyCommNetTask(TaskNode* src, TaskNode* dst);

  std::unique_ptr<const ChainGraph> chain_gph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
