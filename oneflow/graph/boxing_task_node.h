#ifndef ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_BOXING_TASK_NODE_H_

#include "oneflow/graph/comp_task_node.h"

namespace oneflow {

class BoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;
  
  std::string VisualStr() const override {
    return TaskNode::VisualStr() + "Boxing";
  }

  void ToProto(TaskProto* ret) const override {
    TaskNode::ToProto(ret);
  };

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
  OVERRIDE_IF_FW_BP_FOR_FUNC(BuildExecAndEnrollLbn2Regsts);
  OVERRIDE_IF_FW_BP_FOR_FUNC(InferShapeOfBlobsInProducedRegsts);

  void FwBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void FwInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  void BpBuildExecAndEnrollLbn2Regsts(TaskGraph*);
  void BpInferShapeOfBlobsInProducedRegsts(TaskGraph*);
  
  void EnrollAllRegstAndBindRelatedEdge();
  TaskType task_type() const override { return kBoxingTask; }
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
