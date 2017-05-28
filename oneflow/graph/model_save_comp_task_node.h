#ifndef ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
#define ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_

#include "graph/comp_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

  void set_related_update_task_parallel_id(uint64_t parallel_id) {
    related_update_task_parallel_id_ = parallel_id;
  }

  void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
    ret->set_parallel_id(related_update_task_parallel_id_);
  }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  bool IsMeaningLess() const override {
    return !GetSubscribedRegstDesc("model");
  }

  TaskType task_type() const override {
    return kMdSaveCompTask;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdSaveCompTaskNode> ();
  }

  uint64_t related_update_task_parallel_id_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
