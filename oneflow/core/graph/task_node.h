#ifndef ONEFLOW_CORE_GRAPH_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TASK_NODE_H_

#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class TaskEdge;

class TaskNode : public Node<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode();
  virtual ~TaskNode() = default;

  // Getters
  int64_t machine_id() const { return machine_id_; }
  int64_t thrd_id() const { return thrd_id_; }
  int64_t task_id() const { return task_id_; }
  std::shared_ptr<RegstDesc> GetProducedRegst(const std::string& name);
  DeviceType device_type() const;
  virtual const ParallelContext* parallel_ctx() const { return nullptr; }

  // Setters
  void set_machine_id(int64_t val);
  void set_thrd_id(int64_t val);

  // Build
  virtual void ProduceAllRegstsAndBindEdges() { TODO(); }
  virtual void ConsumeAllRegsts() { TODO(); }
  void Build();
  virtual bool IsReadyForBuild() { return IsAllConsumedRegstLocked(); }
  void EraseEmptyProducedRegst();
  void InferMemCaseOfProducedRegst();

  // Others
  virtual TaskType GetTaskType() const = 0;
  std::string VisualStr() const override;
  virtual bool IsMeaningLess();
  virtual void ToProto(TaskProto*);

 protected:
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name,
                                          int32_t min_register_num,
                                          int32_t max_register_num);
  void ConsumeRegst(const std::string& name, std::shared_ptr<RegstDesc>);
  bool IsAllConsumedRegstLocked();
  ExecGraph& mut_exec_gph() { return exec_gph_; }
  std::shared_ptr<RegstDesc> GetConsumedRegst(const std::string& name);

  virtual void BuildExecGphAndRegst() { TODO(); }
  virtual void LockRegsts();

 private:
  void UpdateTaskId();

  int64_t machine_id_;
  int64_t thrd_id_;
  int64_t task_id_;

  ExecGraph exec_gph_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> produced_regsts_;
  HashMap<std::string, std::weak_ptr<RegstDesc>> consumed_regsts_;
};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() = default;

  std::shared_ptr<RegstDesc> GetRegst(
      const std::string& name_in_producer) const;
  void AddRegst(const std::string& name_in_producer,
                std::shared_ptr<RegstDesc> regst);
  std::shared_ptr<RegstDesc> GetSoleRegst() const;

 private:
  HashMap<std::string, std::weak_ptr<RegstDesc>> name_in_producer2regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_NODE_H_
