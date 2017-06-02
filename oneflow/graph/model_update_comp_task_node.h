#ifndef ONEFLOW_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_

#include "oneflow/graph/comp_task_node.h"

namespace oneflow {

class MdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompTaskNode);
  MdUpdtCompTaskNode() = default;
  ~MdUpdtCompTaskNode() = default;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override {
    return kMdUpdtCompTask;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdUpdtCompTaskNode> ();
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
