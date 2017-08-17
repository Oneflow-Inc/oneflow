#ifndef ONEFLOW_CORE_GRAPH_DATA_COMP_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_DATA_COMP_TASK_NODE_H_

#include "oneflow/core/graph/comp_task_node.h"

namespace oneflow {

class DataCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataCompTaskNode);
  DataCompTaskNode() = default;
  ~DataCompTaskNode() = default;

  void ToProto(TaskProto* proto, std::function<int64_t(const ChainNode*)>
                                     MeaninglessTaskCnt4Chain) const override {
    TaskNode::ToProto(proto, MeaninglessTaskCnt4Chain);
    FillProtoWithParallelInfo(proto, MeaninglessTaskCnt4Chain);
  }

  void FillProtoWithParallelInfo(TaskProto* proto,
                                 std::function<int64_t(const ChainNode*)>
                                     MeaninglessTaskCnt4Chain) const override {
    auto parallel_desc = chain_node()->parallel_desc();
    proto->set_parallel_policy(parallel_desc->policy());
    proto->set_parallel_id(parallel_id());
    proto->set_parallel_num(parallel_desc->parallel_num()
                            - MeaninglessTaskCnt4Chain(chain_node()));
  }

  bool IsMeaningLess() const override {
    if (IsFwNode()) {
      if (chain_node()->IsRecordNode()) {
        return false;
      } else {
        return TaskNode::IsMeaningLess();
      }
    } else {
      return TaskNode::IsMeaningLess() || GetFwNode()->IsMeaningLess();
    }
  }

 private:
  OVERRIDE_IF_FW_BP_FOR_FUNC(BuildExecAndEnrollLbn2Regsts);
  OVERRIDE_IF_FW_BP_FOR_FUNC(InferShapeOfBlobsInProducedRegsts);

  using Lbn2NodeBnMap = HashMap<std::string, std::pair<ExecNode*, std::string>>;

  void FwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph);
  void FwInferShapeOfBlobsInProducedRegsts(TaskGraph* gph);
  void FwBuildFromUserOps(Lbn2NodeBnMap* lbn2producer,
                          Lbn2NodeBnMap* extern_in_lbn2consumer);
  void FwSetExecNodeFromInRegst(const Lbn2NodeBnMap& extern_in_lbn2consumer);
  void FwEnrollLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer);
  void FwEnrollLbn2OutRegstWhenLoss();
  void FwEnrollLbn2OutRegstWhenNotLoss(const Lbn2NodeBnMap& lbn2producer);
  void FwEnrollLbn2ActivationRegst();
  void FwEnrollLbn2ModelAndTmpRegsts();
  void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void BpInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void BpBuildExecGraph();
  void BpEnrollLbn2ProducedRegst();
  void BpEnrollLbn2ActivationDiffRegst();
  void BpSetExecNodeFromOutDiffRegst();
  void BpEnrollLbn2InDiffRegst();
  void BpEnrollLbn2ModelDiffRegst();

  TaskType task_type() const override { return kDataCompTask; }
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return of_make_unique<DataCompTaskNode>();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_DATA_COMP_TASK_NODE_H_
