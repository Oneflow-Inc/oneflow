#ifndef ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_UPDATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/model_update_compute_task_node.h"

namespace oneflow {

class EmbeddingLookupMdUpdtCompTaskNode final : public MdUpdtCompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupMdUpdtCompTaskNode);
  EmbeddingLookupMdUpdtCompTaskNode() = default;
  ~EmbeddingLookupMdUpdtCompTaskNode() = default;

  TaskType GetTaskType() const override {
    return TaskType::kEmbeddingLookupMdUpdt;
  }

 private:
  std::shared_ptr<const Operator> ConstructModelUpdateOp(
      int32_t in_num) override;
  void BindInRegst() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_UPDATE_COMPUTE_TASK_NODE_H_
