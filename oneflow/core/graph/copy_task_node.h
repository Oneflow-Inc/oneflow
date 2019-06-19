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
  void InferProducedDataRegstTimeShape() final;
};

class CopyHdTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdTaskNode);
  CopyHdTaskNode() = default;
  ~CopyHdTaskNode() = default;

  TaskType GetTaskType() const override { return TaskType::kCopyHd; }

  void Init(CopyHdOpConf::Type, int64_t machine_id, int64_t dev_phy_id);

  CopyHdOpConf::Type copy_type() const { return copy_type_; }
  int64_t MemZoneId121() const override {
    if (copy_type_ == CopyHdOpConf::H2D) {
      return TaskNode::MemZoneId121();
    } else if (copy_type_ == CopyHdOpConf::D2H) {
      return Global<IDMgr>::Get()->CpuMemZoneId();
    } else if (copy_type_ == CopyHdOpConf::D2D) {
      return TaskNode::MemZoneId121();
    } else {
      UNIMPLEMENTED();
    }
  }

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

  void Init(int64_t machine_id, int64_t src_machine_id);
  int64_t AllocateLocalWorkStreamId() override;
  int64_t peer_machine_id() const { return peer_machine_id_; }

 private:
  void InitProducedRegstMemCase(MemoryCase*) override;
  void PinConsumedRegstMemCase(MemoryCase*) override;
  OperatorConf NewCopyOpConf() override;
  int64_t peer_machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
