#ifndef ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class MdDiffAccCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdDiffAccCompTaskNode);
  MdDiffAccCompTaskNode() = default;
  ~MdDiffAccCompTaskNode() = default;

  void ToProto(TaskProto* proto, std::function<int64_t(const ChainNode*)>
                                     MeaninglessTaskCnt4Chain) const override {
    TaskNode::ToProto(proto, MeaninglessTaskCnt4Chain);
    fw_task_->FillProtoWithParallelInfo(proto, MeaninglessTaskCnt4Chain);
  }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferShapeOfBlobsInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override { return kMdDiffAccCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdDiffAccCompTaskNode>();
  }
  CompTaskNode* fw_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_DIFF_ACCUMULATE_COMP_TASK_NODE_H_
