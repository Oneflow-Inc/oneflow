#ifndef ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class MdDiffAccCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccCompTaskNode);
  MdDiffAccCompTaskNode() = default;
  ~MdDiffAccCompTaskNode() = default;

  void ToProto(TaskProto* proto) const override {
    TaskNode::ToProto(proto);
    proto->set_parallel_policy(chain_node()->parallel_desc()->policy());
    proto->set_parallel_id(parallel_id());
    proto->set_parallel_num(chain_node()->parallel_desc()->parallel_num());
  }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override {
    return kMdDiffAccCompTask;
  }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdDiffAccCompTaskNode> ();
  }

};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_
