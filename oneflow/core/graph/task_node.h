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

RegstDescProto* FindOrCreateProducedCtrlRegstDesc(TaskProto* task_proto,
                                                  const std::string& regst_desc_name);
RegstDescIdSet* FindOrCreateConsumedCtrlRegstDescIdSet(TaskProto* task_proto,
                                                       const std::string& regst_desc_name);
MemoryCase GetDefaultMemCase(const TaskNode* node);

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
  int64_t area_id() const { return area_id_; }
  int64_t chain_id() const { return chain_id_; }
  int64_t order_in_graph() const { return order_in_graph_; }
  const ExecGraph& exec_gph() const { return exec_gph_; }
  std::shared_ptr<RegstDesc> GetProducedRegst(const std::string& name);
  const std::list<std::shared_ptr<RegstDesc>>& GetConsumedRegst(const std::string& name);
  std::shared_ptr<RegstDesc> GetSoleConsumedRegst(const std::string& name);
  const HashMap<std::string, std::shared_ptr<RegstDesc>>& produced_regsts() const {
    return produced_regsts_;
  }
  const HashMap<std::string, std::list<std::shared_ptr<RegstDesc>>>& consumed_regsts() const {
    return consumed_regsts_;
  }
  DeviceType device_type() const;
  virtual const ParallelContext* parallel_ctx() const { return nullptr; }
  int64_t LocalWorkStreamId() const;
  int64_t GlobalWorkStreamId() const;
  int64_t GpuPhyId() const { return Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id_); }
  virtual int64_t AreaId4ChainMerge() const { return area_id(); }

  // Setters
  void set_machine_id(int64_t val);
  void set_thrd_id(int64_t val);
  void set_area_id(int64_t val);
  void set_chain_id(int64_t val);
  void set_order_in_graph(int64_t val);

  // Build
  virtual void ProduceAllRegstsAndBindEdges() = 0;
  virtual void ConsumeAllRegsts() = 0;
  void PinConsumedRegst();
  void InferTimeShapeIfMeaningful();
  void ForEachProducedDataRegst(std::function<void(const std::string&, RegstDesc*)> Handler);
  void ForEachConsumedDataRegst(
      std::function<void(const std::string&, const RegstDesc*)> Handler) const;
  void Build();
  virtual bool IsReadyForBuild() { return IsAllConsumedDataRegstLocked(); }

  void EraseZeroSizeProducedBlob();
  void EraseZeroSizeConsumedRegst();
  void EraseZeroSizeProducedRegst();
  void UnbindBnWithEmptyRegst();

  // Others
  virtual TaskType GetTaskType() const { return TaskType::kInvalid; }
  std::string VisualStr() const override;
  virtual bool IsMeaningLess();
  virtual void ToProto(TaskProto*);
  virtual bool IsPersistence() const { return false; }
  void BindEdgeWithProducedRegst(TaskEdge*, const std::string& name);
  virtual int64_t MemZoneId121() const;  // TODO: there is bug for reduce task node
  void BuildCtrlRegstDescIfNeed(TaskNode* dst_node);
  RegstDesc* BuildCtrlRegstDesc(TaskNode* dst_node);
  RegstDesc* BuildCtrlRegstDesc(TaskNode* dst_node, std::string* name);

  void ForEachInDataEdge(const std::function<void(TaskEdge*)>& Handler) const;
  void ForEachOutDataEdge(const std::function<void(TaskEdge*)>& Handler) const;

  void ForEachNodeOnInDataEdge(const std::function<void(TaskNode*)>& Handler) const;
  void ForEachNodeOnOutDataEdge(const std::function<void(TaskNode*)>& Handler) const;
  void ForEachNodeOnInOutDataEdge(const std::function<void(TaskNode*)>& Handler) const;

  TaskEdge* SoleInDataEdge() const;
  TaskEdge* SoleOutDataEdge() const;
  size_t in_data_edges_size() const;
  size_t out_data_edges_size() const;

 protected:
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_mem_sharing);
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_mem_sharing,
                                          int32_t min_register_num, int32_t max_register_num);
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_mem_sharing,
                                          int32_t min_register_num, int32_t max_register_num,
                                          const RegstDescTypeProto&);
  std::shared_ptr<RegstDesc> NewProducedRegst(bool enable_mem_sharing, int32_t min_register_num,
                                              int32_t max_register_num, const RegstDescTypeProto&);
  virtual void InitProducedRegstMemCase(RegstDesc* regst);
  virtual void InitProducedRegstMemCase(MemoryCase*);
  virtual void PinConsumedRegstMemCase(MemoryCase*);
  void ConsumeRegst(const std::string& name, std::shared_ptr<RegstDesc>);
  bool IsAllConsumedDataRegstLocked();
  ExecGraph& mut_exec_gph() { return exec_gph_; }
  void TryLockConsumedRegst(const std::string& name);
  void EraseConsumedRegstsByName(const std::string& name);

  virtual void BuildExecGphAndRegst() = 0;
  virtual void LockRegsts();
  void FixRegisterNumRange();
  virtual void FixPackedBlobDescOfProducedRegst() {}

  virtual int64_t AllocateLocalWorkStreamId();

  virtual void InferProducedDataRegstTimeShape() = 0;
  void NaiveInferProducedDataRegstTimeShape();

  TaskEdge* GetSoleEdge(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                            const) const;
  size_t GetEdgesSize(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                          const) const;

 private:
  void UpdateTaskId();

  int64_t machine_id_;
  int64_t thrd_id_;
  int64_t task_id_;
  int64_t area_id_;
  int64_t chain_id_;
  int64_t order_in_graph_;

  ExecGraph exec_gph_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> produced_regsts_;
  HashMap<std::string, std::list<std::shared_ptr<RegstDesc>>> consumed_regsts_;
};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() = default;

  std::shared_ptr<RegstDesc> GetRegst(const std::string& name_in_producer) const;
  std::shared_ptr<RegstDesc> GetSoleRegst() const;
  std::vector<std::shared_ptr<RegstDesc>> GetRegsts() const;

  void AddRegst(const std::string& name_in_producer, std::shared_ptr<RegstDesc> regst);

 private:
  HashMap<std::string, std::shared_ptr<RegstDesc>> name_in_producer2regst_;
};

extern std::map<TaskType, std::string> task_type2color;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_NODE_H_
