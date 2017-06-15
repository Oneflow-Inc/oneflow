#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class MdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompTaskNode);
  MdUpdtCompTaskNode() = default;
  ~MdUpdtCompTaskNode() = default;
  
  void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    // TODO
    // parallel_policy
    // parallel_id
    // parallel_num
  }

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

#endif // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
