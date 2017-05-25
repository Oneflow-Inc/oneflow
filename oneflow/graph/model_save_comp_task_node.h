#ifndef ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_

#include "graph/comp_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph);
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph);
  bool IsMeaningLess() const override {
    return !GetSubscribedRegstDesc("model");
  }

  TaskType task_type() const override {
    return kMdSaveCompTask;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdSaveCompTaskNode> ();
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
