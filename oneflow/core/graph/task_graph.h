#ifndef ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_

#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/compute_task_node.h"
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

  void BldSubTskGphByNormalBoxing(const ChainNode* src, const ChainNode* dst);
  void BldSubTskGphByAddCloneBoxing(const ChainNode* src, const ChainNode* dst);
  void BldSubTskGphByDirectOneToOne(const ChainNode* src, const ChainNode* dst);
  void BldSubTskGphByInDirectOneToOne(const ChainNode* src,
                                      const ChainNode* dst);
  void BldSubTskGphBySelectOneSourceToSoleSink(const ChainNode* src,
                                               const ChainNode* dst);

 private:
  std::unique_ptr<const ChainGraph> chain_gph_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_GRAPH_H_
