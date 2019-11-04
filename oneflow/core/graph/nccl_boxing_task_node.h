#ifndef ONEFLOW_CORE_GRAPH_NCCL_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_NCCL_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class NcclBoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingTaskNode);
  NcclBoxingTaskNode() = default;
  ~NcclBoxingTaskNode() override = default;

  void Init(int64_t machine_id, int64_t dev_phy_id, const ParallelContext& parallel_ctx,
            const LogicalBlobId& lbi);

 protected:
  virtual std::shared_ptr<Operator> NewNcclBoxingOp() const = 0;
  const LogicalBlobId& GetLbi() const { return lbi_; }

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;
  void ToProto(TaskProto*) override;
  const ParallelContext* parallel_ctx() const override { return &parallel_ctx_; }

  ParallelContext parallel_ctx_;
  LogicalBlobId lbi_;
};

class NcclBoxingReduceScatterTaskNode final : public NcclBoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingReduceScatterTaskNode);
  NcclBoxingReduceScatterTaskNode() = default;
  ~NcclBoxingReduceScatterTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclBoxingReduceScatter; }
  std::shared_ptr<Operator> NewNcclBoxingOp() const override;
};

class NcclBoxingAllGatherTaskNode final : public NcclBoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingAllGatherTaskNode);
  NcclBoxingAllGatherTaskNode() = default;
  ~NcclBoxingAllGatherTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclBoxingAllGather; }
  std::shared_ptr<Operator> NewNcclBoxingOp() const override;
};

class NcclBoxingAllReduceTaskNode final : public NcclBoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingAllReduceTaskNode);
  NcclBoxingAllReduceTaskNode() = default;
  ~NcclBoxingAllReduceTaskNode() override = default;

  TaskType GetTaskType() const override { return TaskType::kNcclBoxingAllReduce; }
  std::shared_ptr<Operator> NewNcclBoxingOp() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_NCCL_BOXING_TASK_NODE_H_
