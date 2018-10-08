#ifndef ONEFLOW_CORE_GRAPH_PIPE_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_PIPE_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class PipeCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PipeCompTaskNode);
  PipeCompTaskNode() = default;
  ~PipeCompTaskNode() override = default;

 private:
  void BuildExecGphAndRegst() override;
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() final;
  void InferProducedDataRegstTimeShape() final;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_PIPE_COMPUTE_TASK_NODE_H_
