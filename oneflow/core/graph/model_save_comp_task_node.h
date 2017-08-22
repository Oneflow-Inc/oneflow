#ifndef ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class MdSaveCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdSaveCompTaskNode);
  MdSaveCompTaskNode() = default;
  ~MdSaveCompTaskNode() = default;

  void ToProto(TaskProto* proto, std::function<int64_t(const ChainNode*)>
                                     MeaninglessTaskCnt4Chain) const override {
    TaskNode::ToProto(proto, MeaninglessTaskCnt4Chain);
    fw_task_->FillProtoWithParallelInfo(proto, MeaninglessTaskCnt4Chain);
  }

  void set_fw_task(CompTaskNode* fw_task) { fw_task_ = fw_task; }
  CompTaskNode* fw_task() { return fw_task_; }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferBlobDescInProducedRegsts(TaskGraph* gph) override;
  bool IsMeaningLess() const override { return !GetConsumedRegstDesc("model"); }

  TaskType task_type() const override { return kMdSaveCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdSaveCompTaskNode>();
  }
  CompTaskNode* fw_task_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_SAVE_COMP_TASK_NODE_H_
