#ifndef ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_BOXING_TASK_NODE_H_

#include "graph/comp_task_node.h"

namespace oneflow {

class BoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;
  
  std::string VisualStr() const override {
    return "Boxing_" + node_id_str();
  }

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
  
  using ChainEdgesPair =
      std::pair<const ChainNode*, std::vector<const TaskEdge*>>;
  using Chain2EdgesMap =
      HashMap<const ChainNode*, std::vector<const TaskEdge*>>;
  void FwInitChain2SortedEdgesMaps(
      Chain2EdgesMap* chain2sorted_edges,
      const std::unordered_set<TaskEdge*>& (TaskNode::*in_out_edges)() const,
      TaskNode* (TaskEdge::*src_dst_node)() const,
      TaskEdge* (TaskNode::*SoleEdge)() const);
  void FwSortEdgesInnerStage(
      std::vector<const TaskEdge*>* edges_to_be_sorted,
      TaskNode* (TaskEdge::*src_dst_node)() const,
      TaskEdge* (TaskNode::*SoleEdge)() const);
  void FwBuildChainSortedEdgesPair(
      const ChainEdgesPair& chain_sorted_in_edges,
      const ChainEdgesPair& chain_sorted_out_edges);
  virtual void FwVirtualBuild() = 0;

 private:
  void FwBuildExecAndProducedRegsts(TaskGraph*) override;
  void BpBuildExecAndProducedRegsts(TaskGraph*) override;
  
  void EnrollAllRegstAndBindRelatedEdge();
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
