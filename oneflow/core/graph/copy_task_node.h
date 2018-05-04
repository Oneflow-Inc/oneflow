#ifndef ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_

#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class CopyTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 protected:
  virtual OperatorConf NewCopyOpConf() = 0;

 private:
};

class CopyHdTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdTaskNode);
  CopyHdTaskNode() = default;
  ~CopyHdTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyHd; }

  void Init(int64_t machine_id, int64_t thrd_id, CopyHdOpConf::Type);

  CopyHdOpConf::Type copy_type() const { return copy_type_; }

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;

  CopyHdOpConf::Type copy_type_;
};

class CopyCommNetTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetTaskNode);
  CopyCommNetTaskNode() = default;
  ~CopyCommNetTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyCommNet; }

  void Init(int64_t machine_id);

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  void PinConsumedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
