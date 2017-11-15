#ifndef ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class CopyTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyTaskNode);
  CopyTaskNode() = default;
  virtual ~CopyTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildRegsts() override;

 private:
};

class CopyHdTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdTaskNode);
  CopyHdTaskNode() = default;
  ~CopyHdTaskNode() = default;

  TodoTaskType GetTaskType() const override { return TodoTaskType::kCopyHd; }

  void Init(const CompTaskNode*, CopyHdOpConf::Type);

 private:
};

class CopyCommNetTaskNode final : public CopyTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetTaskNode);
  CopyCommNetTaskNode() = default;
  ~CopyCommNetTaskNode() = default;

  TodoTaskType GetTaskType() const override {
    return TodoTaskType::kCopyCommNet;
  }

  void Init(int64_t machine_id);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_COPY_TASK_NODE_H_
