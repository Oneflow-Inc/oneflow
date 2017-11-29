#ifndef ONEFLOW_CORE_GRAPH_SINK_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_SINK_COMPUTE_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SinkCompTaskNode : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SinkCompTaskNode);
  SinkCompTaskNode() = default;
  virtual ~SinkCompTaskNode() = default;

 private:
  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

  void FixThrdId() override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SINK_COMPUTE_TASK_NODE_H_
