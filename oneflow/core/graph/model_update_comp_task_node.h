#ifndef ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_

#include "oneflow/core/graph/data_comp_task_node.h"

namespace oneflow {

class MdUpdtCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompTaskNode);
  MdUpdtCompTaskNode() = default;
  ~MdUpdtCompTaskNode() = default;

  void ToProto(TaskProto* proto, std::function<int64_t(const ChainNode*)>
                                     MeaninglessTaskCnt4Chain) const override {
    TaskNode::ToProto(proto, MeaninglessTaskCnt4Chain);
    fw_task_->FillProtoWithParallelInfo(proto, MeaninglessTaskCnt4Chain);
    int64_t related_save_task_id = -1;
    for (const auto& pair : produced_regst_descs()) {
      for (const TaskNode* consumer : pair.second->consumers()) {
        if (dynamic_cast<const DataCompTaskNode*>(consumer) == nullptr) {
          CHECK_EQ(related_save_task_id, -1);
          related_save_task_id = consumer->task_id();
        }
      }
    }
    proto->set_related_save_task_id(related_save_task_id);
    proto->set_random_seed(random_seed_);
  }

  void set_fw_task(CompTaskNode* fw_task) { fw_task_ = fw_task; }
  CompTaskNode* fw_task() { return fw_task_; }

  void set_random_seed(uint32_t val) { random_seed_ = val; }

 private:
  void BuildExecAndEnrollLbn2Regsts(TaskGraph* gph) override;
  void InferBlobDescInProducedRegsts(TaskGraph* gph) override;
  TaskType task_type() const override { return kMdUpdtCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<MdUpdtCompTaskNode>();
  }
  CompTaskNode* fw_task_;
  uint32_t random_seed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_MODEL_UPDATE_COMP_TASK_NODE_H_
