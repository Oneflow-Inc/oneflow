#ifndef ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_GRAPH_BOXING_TASK_NODE_H_

#include "graph/task_node.h"

namespace oneflow {

class BoxingTaskNode : public TaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;

 protected:
  virtual void InitWithFwNode(TaskNode* fw_node) override {
    TaskNode::InitWithFwNode(fw_node);
  }
  
  using OpPair = std::pair<std::shared_ptr<Operator>, std::shared_ptr<Operator>>;
  static OpPair FwBuildBoxingOpDataData();
  static OpPair FwBuildBoxingOpDataModel();
  static OpPair FwBuildBoxingOpModelData();
  static OpPair FwBuildBoxingOpModelModel();
  
  using Chain2EdgesMap =
      std::unordered_map<const ChainNode*, std::vector<const TaskEdge*>>;
  void SetOutEdgeRegisterPtr();
  void FwInitChain2SortedEdgesMaps(
      Chain2EdgesMap* chain2sorted_edges,
      const std::unordered_set<TaskEdge*>& (TaskNode::*in_out_edges)() const,
      TaskNode* (TaskEdge::*src_dst_node)() const,
      TaskEdge* (TaskNode::*SoleEdge)() const);

 private:
  
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BOXING_TASK_NODE_H_
