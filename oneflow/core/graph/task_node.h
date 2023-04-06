/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_GRAPH_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TASK_NODE_H_

#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/memory/memory_zone.h"

namespace std {

template<>
struct hash<oneflow::TaskType> {
  std::size_t operator()(const oneflow::TaskType& task_type) const {
    return std::hash<uint32_t>{}(static_cast<uint32_t>(task_type));
  }
};

}  // namespace std

namespace oneflow {

RegstDescProto* FindOrCreateProducedCtrlRegstDesc(TaskProto* task_proto,
                                                  const std::string& regst_desc_name);
RegstDescIdSet* FindOrCreateConsumedCtrlRegstDescIdSet(TaskProto* task_proto,
                                                       const std::string& regst_desc_name);

bool inline IsValidChainId(int64_t val) { return val >= 0; }

class TaskEdge;

class TaskNode : public Node<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode();
  ~TaskNode() override = default;

  // Getters
  int64_t machine_id() const { return machine_id_; }
  int64_t thrd_id() const { return thrd_id_; }
  int64_t task_id() const { return task_id_; }
  const StreamId& stream_id() const;
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

  // Setters
  void set_machine_id(int64_t val);
  void set_thrd_id(int64_t val);
  void set_chain_id(int64_t val);
  void set_order_in_graph(int64_t val);

  // Build
  virtual void ProduceAllRegstsAndBindEdges() = 0;
  virtual void ConsumeAllRegsts() = 0;
  void PinConsumedRegst();
  void InferTimeShapeIfMeaningful();
  void ForEachProducedDataRegst(const std::function<void(const std::string&, RegstDesc*)>& Handler);
  void ForEachConsumedDataRegst(
      const std::function<void(const std::string&, const RegstDesc*)>& Handler) const;
  void Build();

  void EraseUninitializedShapeProducedBlob();
  void EraseZeroSizeConsumedRegst();
  void EraseZeroSizeProducedRegst();
  void UnbindBnWithEmptyRegst();

  // Others
  virtual TaskType GetTaskType() const { return TaskType::kInvalid; }
  std::string VisualStr() const override;
  virtual bool IsMeaningLess();
  virtual void ToProto(TaskProto*) const;
  void BindEdgeWithProducedRegst(TaskEdge*, const std::string& name);
  virtual MemZoneId MemZoneId121() const;
  bool BuildCtrlRegstDescIfNeed(TaskNode* dst_node, std::string* name);
  RegstDesc* BuildCtrlRegstDesc(TaskNode* dst_node);
  RegstDesc* BuildCtrlRegstDesc(TaskNode* dst_node, std::string* name);
  std::shared_ptr<Shape> GetFastestInputOutputTimeShape() const;

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
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_reuse_mem);
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                          int32_t min_register_num, int32_t max_register_num);
  std::shared_ptr<RegstDesc> ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                          int32_t min_register_num, int32_t max_register_num,
                                          const RegstDescTypeProto&);
  std::shared_ptr<RegstDesc> NewProducedRegst(bool enable_reuse_mem, int32_t min_register_num,
                                              int32_t max_register_num, const RegstDescTypeProto&);
  virtual void InitProducedRegstMemCase(RegstDesc* regst);
  virtual void InitProducedRegstMemCase(MemoryCase*);
  virtual void PinConsumedRegstMemCase(MemoryCase*);
  void ConsumeRegst(const std::string& name);
  void ConsumeRegst(const std::string& name, const std::shared_ptr<RegstDesc>&);
  ExecGraph& mut_exec_gph() { return exec_gph_; }
  void EraseConsumedRegstsByName(const std::string& name);

  virtual void BuildExecGphAndRegst() = 0;

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
  int64_t chain_id_;
  int64_t order_in_graph_;
  std::unique_ptr<TaskId> new_task_id_;

  ExecGraph exec_gph_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> produced_regsts_;
  HashMap<std::string, std::list<std::shared_ptr<RegstDesc>>> consumed_regsts_;
};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() = default;
  ~TaskEdge() override = default;

  std::shared_ptr<RegstDesc> GetRegst(const std::string& name_in_producer) const;
  std::shared_ptr<RegstDesc> GetSoleRegst() const;
  std::vector<std::shared_ptr<RegstDesc>> GetRegsts() const;
  const HashSet<LogicalBlobId>& GetLbis() const { return lbis_; }

  void AddRegst(const std::string& name_in_producer, const std::shared_ptr<RegstDesc>& regst);
  void AddLbi(const LogicalBlobId& lbi) { lbis_.insert(lbi); }
  void AddLbis(const std::vector<LogicalBlobId>& lbis) { lbis_.insert(lbis.begin(), lbis.end()); }

  void CheckRegstLbiValid() const;

 private:
  HashSet<LogicalBlobId> lbis_;
  HashMap<std::string, std::shared_ptr<RegstDesc>> name_in_producer2regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_NODE_H_
