#ifndef ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class EmbeddingLookupMdDiffAccCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupMdDiffAccCompTaskNode);
  EmbeddingLookupMdDiffAccCompTaskNode() = default;
  ~EmbeddingLookupMdDiffAccCompTaskNode() = default;
  TaskType GetTaskType() const override {
    return TaskType::kEmbeddingLookupMdDiffAcc;
  }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_EMBEDDING_LOOKUP_MODEL_DIFF_ACCUMULATE_COMPUTE_TASK_NODE_H_
