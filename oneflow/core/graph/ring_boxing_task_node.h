#ifndef ONEFLOW_CORE_GRAPH_RING_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_RING_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

enum RingBoxingTaskMode {
  kRingBoxingTaskModeInvalid,
  kRingBoxingTaskModeP2S,
  kRingBoxingTaskModeS2B,
  kRingBoxingTaskModeP2B
};

class RingBoxingTaskNode final : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingBoxingTaskNode);
  RingBoxingTaskNode() = default;
  ~RingBoxingTaskNode() override = default;

  void Init(RingBoxingTaskMode mode, int64_t machine_id, int64_t thrd_id, const LogicalBlobId& lbi,
            const Shape& logical_blob_shape, TaskNode* send_to, TaskNode* recv_from,
            const std::vector<TensorSliceView>& slices, const std::vector<int64_t>& ring,
            const ParallelContext& parallel_ctx);
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  TaskType GetTaskType() const override { return TaskType::kRingBoxing; }
  void SetOutShape(const Shape& shape);

 private:
  bool IsReadyForBuild() override;
  void BuildExecGphAndRegst() override;
  void InferProducedDataRegstTimeShape() override;
  OperatorConf GetBoxingOpConf() const;
  const ParallelContext* parallel_ctx() const override { return &parallel_ctx_; }
  void ToProto(TaskProto*) override;

  LogicalBlobId lbi_;
  Shape logical_blob_shape_;
  RingBoxingTaskMode mode_ = kRingBoxingTaskModeInvalid;
  std::vector<int64_t> ring_;
  std::unique_ptr<Shape> out_shape_;
  std::vector<TensorSliceView> slices_;
  ParallelContext parallel_ctx_;
  TaskNode* send_to_;
  TaskNode* recv_from_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_RING_BOXING_TASK_NODE_H_
