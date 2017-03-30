#ifndef ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_

#include "graph/boxing_task_node.h"

namespace oneflow {

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

 private:
  std::unique_ptr<TaskNode> CreateSameTypeNode() const override {
    return std::unique_ptr<TaskNode> (new InBoxingTaskNode);
  }
  void InitWithFwNode(TaskNode* fw_node) override {
    BoxingTaskNode::InitWithFwNode(fw_node);
  }

  using ChainEdgesPair =
      std::pair<const ChainNode*, std::vector<const TaskEdge*>>;
  using Chain2EdgesMap =
      std::unordered_map<const ChainNode*, std::vector<const TaskEdge*>>;
  void FwBuildExecGraphAndSetProducedRegisterDescs() override;
  void SetOutEdgeRegisterPtr();
  void FwInitChain2SortedInEdgesMaps(Chain2EdgesMap* chain2sorted_in_edges);
  void FwInitSortedOutEdges(std::vector<const TaskEdge*>* sorted_out_edges);
  void FwBuildChainSortedEdgesPair(
      const ChainEdgesPair& chain_sorted_in_edges,
      const std::vector<const TaskEdge*>& sorted_out_edges);
  void FwSetProducedRegister();
  void BpBuildExecGraphAndSetProducedRegisterDescs() override;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_IN_BOXING_TASK_NODE_H_
