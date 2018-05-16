#ifndef ONEFLOW_CORE_GRAPH_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TASK_NODE_H_

#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

bool IsForwardTaskType(TaskType);
bool IsBackwardTaskType(TaskType);
bool IsMdUpdtTaskType(TaskType);

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
  const ExecGraph& exec_gph() const { return exec_gph_; }
  std::shared_ptr<RegstDesc> GetProducedRegst(const std::string& name);
  const std::list<std::weak_ptr<RegstDesc>>& GetConsumedRegst(const std::string& name);
  std::shared_ptr<RegstDesc> GetSoleConsumedRegst(const std::string& name);
  DeviceType device_type() const;
  virtual const ParallelContext* parallel_ctx() const { return nullptr; }
  int64_t MemZoneId() const { return Global<IDMgr>::Get()->GetMemZoneIdFromThrdId(thrd_id_); }

  // Setters
  void set_machine_id(int64_t val);
  void set_thrd_id(int64_t val);

  // Build
  virtual void ProduceAllRegstsAndBindEdges() = 0;
  virtual void ConsumeAllRegsts() = 0;
  void PinConsumedRegst();
  void Build();
  virtual bool IsReadyForBuild() { return IsAllConsumedRegstLocked(); }
  virtual void EraseEmptyProducedRegst();

  // Others
  virtual TaskType GetTaskType() const { return TaskType::kInvalid; }
  std::string VisualStr() const override;
  virtual bool IsMeaningLess();
  virtual void ToProto(TaskProto*);
  virtual bool IsPersistence() const { return false; }
  void BindEdgeWithProducedRegst(TaskEdge*, const std::string& name);

 protected:
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name);
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, int32_t min_register_num,
                                          int32_t max_register_num);
  virtual void InitProducedRegstMemCase(RegstDesc* regst);
  virtual void InitProducedRegstMemCase(MemoryCase*);
  virtual void PinConsumedRegstMemCase(MemoryCase*);
  void ConsumeRegst(const std::string& name, std::shared_ptr<RegstDesc>);
  bool IsAllConsumedRegstLocked();
  ExecGraph& mut_exec_gph() { return exec_gph_; }
  const HashMap<std::string, std::list<std::weak_ptr<RegstDesc>>>& consumed_regsts() {
    return consumed_regsts_;
  }
  void TryLockConsumedRegst(const std::string& name);

  virtual void BuildExecGphAndRegst() = 0;
  virtual void LockRegsts();
  virtual void FixRegisterNumRange();

 private:
  void UpdateTaskId();
  void ClearOutOfDateConsumedRegst();

  int64_t machine_id_;
  int64_t thrd_id_;
  int64_t task_id_;

  ExecGraph exec_gph_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> produced_regsts_;
  HashMap<std::string, std::list<std::weak_ptr<RegstDesc>>> consumed_regsts_;
};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() = default;

  std::shared_ptr<RegstDesc> GetRegst(const std::string& name_in_producer) const;
  std::shared_ptr<RegstDesc> GetSoleRegst() const;

  void AddRegst(const std::string& name_in_producer, std::shared_ptr<RegstDesc> regst);

 private:
  HashMap<std::string, std::weak_ptr<RegstDesc>> name_in_producer2regst_;
};

extern std::map<TaskType, std::string> task_type2color;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_NODE_H_
