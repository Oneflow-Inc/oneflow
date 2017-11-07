#ifndef ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class ChainNode;

class BoxingTaskNode : public TaskNode {
 public:
  struct EdgeInfo {
    const TaskEdge* edge;
    int64_t parallel_id_min;
    int64_t parallel_id_max;
  };

  OF_DISALLOW_COPY_AND_MOVE(BoxingTaskNode);
  BoxingTaskNode() = default;
  virtual ~BoxingTaskNode() = default;

  void Init(int64_t machine_id);
  TodoTaskType GetTaskType() const override { return TodoTaskType::kBoxing; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void Build() override;

  std::shared_ptr<Operator> BldBoxingOpWithAddClone(
      const std::vector<BoxingTaskNode::EdgeInfo>& sorted_in_edges,
      const std::vector<BoxingTaskNode::EdgeInfo>& sorted_out_edges,
      int64_t* used_in_edge_begin, int64_t* used_out_edge_begin);

 private:
  void InitChain2SortedEdgeInfo(
      const std::unordered_set<TaskEdge*>& (TaskNode::*GetEdges)() const,
      TaskEdge* (TaskNode::*SoleEdge)() const,
      TaskNode* (TaskEdge::*SoleNode)() const,
      HashMap<const ChainNode*, std::vector<EdgeInfo>>*);
  void BuildWithChainPair(const ChainNode* in_chain,
                          const std::vector<EdgeInfo>& sorted_in_edges,
                          const ChainNode* out_chain,
                          const std::vector<EdgeInfo>& sorted_out_edges);
  std::shared_ptr<Operator> NewBoxingOp(
      const ChainNode* in_chain, const ChainNode* out_chain,
      const std::vector<EdgeInfo>& sorted_in_edges,
      const std::vector<EdgeInfo>& sorted_out_edges,
      int64_t* used_in_edge_begin, int64_t* used_out_edge_begin);
};

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

 private:
};

class OutBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutBoxingTaskNode);
  OutBoxingTaskNode() = default;
  ~OutBoxingTaskNode() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
