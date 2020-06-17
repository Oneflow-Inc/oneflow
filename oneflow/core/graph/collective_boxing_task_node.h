#ifndef ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CollectiveBoxingGenericTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericTaskNode);
  CollectiveBoxingGenericTaskNode() = default;
  ~CollectiveBoxingGenericTaskNode() override = default;

  void Init(int64_t machine_id, int64_t thrd_id, int64_t area_id, const OperatorConf& op_conf);

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;
  TaskType GetTaskType() const override { return TaskType::kCollectiveBoxingGeneric; }

  OperatorConf op_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COLLECTIVE_BOXING_TASK_NODE_H_
