#ifndef ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_BOXING_TASK_NODE_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class LogicalNode;

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

  TaskType GetTaskType() const override { return TaskType::kBoxing; }

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

#define DECLARE_BLD_BOXING_OP_CONF_METHOD(x)                                        \
  void BldBoxingOpConfWith##x(                                                      \
      const LogicalBlobId& lbi, const std::vector<EdgeInfo>& sorted_in_edges,       \
      const LogicalNode* in_logical, const std::vector<EdgeInfo>& sorted_out_edges, \
      const LogicalNode* out_logical, BoxingOpConf*)

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
  DECLARE_BLD_BOXING_OP_CONF_METHOD(PartialTick2SinkTick);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(FwSbpParallel);
  DECLARE_BLD_BOXING_OP_CONF_METHOD(BwSbpParallel);

 private:
  void InitLogical2SortedEdgeInfo(
      void (TaskNode::*ForEachDataEdge)(const std::function<void(TaskEdge*)>&) const,
      TaskEdge* (TaskNode::*SoleEdge)() const, TaskNode* (TaskEdge::*SoleNode)() const,
      HashMap<const LogicalNode*, std::vector<EdgeInfo>>* logical2sorted_edge_info);
  void BuildWithLogicalPair(const LogicalNode* in_logical,
                            const std::vector<EdgeInfo>& sorted_in_edges,
                            const LogicalNode* out_logical,
                            const std::vector<EdgeInfo>& sorted_out_edges);
  std::shared_ptr<Operator> NewBoxingOp(const LogicalBlobId& lbi, const LogicalNode* in_logical,
                                        const LogicalNode* out_logical,
                                        const std::vector<EdgeInfo>& sorted_in_edges,
                                        const std::vector<EdgeInfo>& sorted_out_edges);
  void InferProducedDataRegstTimeShape() final;
};

#define OVERRIDE_BLD_BOXING_OP_METHOD(x) DECLARE_BLD_BOXING_OP_CONF_METHOD(x) override

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
