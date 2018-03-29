#ifndef ONEFLOW_CORE_GRAPH_GATHER_BACKWARD_COMPUTE_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_GATHER_BACKWARD_COMPUTE_TASK_NODE_H_

class GatherForwardCompTaskNode final : public CompTaskNode {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherForwardCompTaskNode);
  GatherForwardCompTaskNode() = default;
  ~GatherForwardCompTaskNode() = default;

  void ProduceAllRegstsAndBindEdges() override;
  void ConsumeAllRegsts() override;
  void BuildExecGphAndRegst() override;

  TaskType GetTaskType() const override { return TaskType::kGatherForward; }
 private:
};

#endif  // ONEFLOW_CORE_GRAPH_GATHER_BACKWARD_COMPUTE_TASK_NODE_H_
