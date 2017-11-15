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

#define DECLARE_BLD_BOXING_OP_CONF_METHOD(x)                                  \
  void BldBoxingOpConfWith##x(                                                \
      const std::string& lbn, const std::vector<EdgeInfo>& sorted_in_edges,   \
      int64_t in_parallel_num, const std::vector<EdgeInfo>& sorted_out_edges, \
      int64_t out_parallel_num, BoxingOpConf*)

#define DECLARE_VIRTUAL_BLD_BOXING_OP_CONF_METHOD(x) \
  virtual DECLARE_BLD_BOXING_OP_CONF_METHOD(x) = 0

  DECLARE_VIRTUAL_BLD_BOXING_OP_CONF_METHOD(DataConcatAndDataSplit);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(DataConcatAndClone);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(DataConcatAndModelSplit);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(ModelConcatAndDataSplit);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(ModelConcatAndClone);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(AddAndDataSplit);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(AddAndModelSplit);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(AddAndClone);

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
      const std::string& lbn, const ChainNode* in_chain,
      const ChainNode* out_chain, const std::vector<EdgeInfo>& sorted_in_edges,
      const std::vector<EdgeInfo>& sorted_out_edges);
};

#define OVERRIDE_BLD_BOXING_OP_METHOD(x) \
  DECLARE_BLD_BOXING_OP_CONF_METHOD(x) override

class InBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InBoxingTaskNode);
  InBoxingTaskNode() = default;
  ~InBoxingTaskNode() = default;

  OVERRIDE_BLD_BOXING_OP_METHOD(DataConcatAndDataSplit);

 private:
};

class OutBoxingTaskNode final : public BoxingTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutBoxingTaskNode);
  OutBoxingTaskNode() = default;
  ~OutBoxingTaskNode() = default;

  OVERRIDE_BLD_BOXING_OP_METHOD(DataConcatAndDataSplit);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
