#ifndef ONEFLOW_CORE_GRAPH_LOCAL_RING_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_LOCAL_RING_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

enum LocalRingBoxingTaskMode {
  kLocalRingBoxingTaskModeInvalid,
  kLocalRingBoxingTaskModeP2S,
  kLocalRingBoxingTaskModeS2B,
  kLocalRingBoxingTaskModeP2B
};

class LocalRingBoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalRingBoxingTaskNode);
  LocalRingBoxingTaskNode() = default;
  ~LocalRingBoxingTaskNode() override = default;

  void Init(int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
            LocalRingBoxingTaskNode* lhs, std::vector<int64_t> ring);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kLocalRingBoxing; }

 private:
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  std::shared_ptr<RegstDesc> send_regst_desc() { return send_regst_desc_; }

  LogicalBlobId lbi_;
  LocalRingBoxingTaskMode mode_ = kLocalRingBoxingTaskModeInvalid;
  std::shared_ptr<RegstDesc> send_regst_desc_;
  LocalRingBoxingTaskNode* lhs_task_node_ = nullptr;
  std::vector<int64_t> ring_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_LOCAL_RING_BOXING_TASK_NODE_H_
