#ifndef ONEFLOW_CORE_GRAPH_TASK_NODE_H_
#define ONEFLOW_CORE_GRAPH_TASK_NODE_H_

#include "oneflow/core/graph/exec_graph.h"
#include "oneflow/core/graph/stage_graph.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

class TaskGraph;
class TaskEdge;

class TaskNode : public Node<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskNode);
  TaskNode();
  virtual ~TaskNode() = default;

  // Getters
  bool IsFwNode() const { return is_fw_node_; }
  bool IsBpNode() const { return !is_fw_node_; }
  TaskNode* GetFwNode() const;
  TaskNode* GetBpNode() const;
  const ChainNode* chain_node() const { return stage_node_->chain_node(); }
  const StageNode* stage_node() const { return stage_node_; }
  const int64_t& thrd_loc_id() const { return thrd_loc_id_; }
  std::string thrd_loc_id_str() const { return std::to_string(thrd_loc_id_); }
  const ExecGraph& exec_gph() const { return exec_gph_; }
  int64_t task_id() const { return task_id_; }
  std::string task_id_str() const { return std::to_string(task_id_); }
  virtual bool IsMeaningLess() const { return produced_regst_descs_.empty(); }
  virtual DeviceType GetDeviceType() const {
    return chain_node()->parallel_desc()->device_type();
  }

  // Setters
  void SetFwNode() { is_fw_node_ = true; }
  void set_stage_node(const StageNode*);
  int64_t& mut_thrd_loc_id();
  void set_task_id();

  // return bp_node
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();

  //
  virtual void BuildExecAndEnrollLbn2Regsts(TaskGraph*) = 0;
  virtual void InferBlobDescInProducedRegsts(TaskGraph*) = 0;

#define OVERRIDE_IF_FW_BP_FOR_FUNC(func_name) \
  void func_name(TaskGraph* gph) override {   \
    if (IsFwNode()) {                         \
      return Fw##func_name(gph);              \
    } else {                                  \
      return Bp##func_name(gph);              \
    }                                         \
  }

  //
  std::shared_ptr<RegstDesc> GetProducedRegstDesc(
      const std::string& regst_desc_name);
  std::shared_ptr<RegstDesc> GetConsumedRegstDesc(
      const std::string& regst_desc_name) const;
  void TakeOverRegstDesc(TaskNode* rhs, const std::string& regst_desc_name);
  void EraseProducedEmptyRegsts();
  void EraseZeroSizeBlobInProducedRegsts();

  const HashMap<std::string, std::shared_ptr<RegstDesc>>& produced_regst_descs()
      const {
    return produced_regst_descs_;
  }

  //
  const TaskEdge* GetOutEdge4ProducedRegst(std::weak_ptr<RegstDesc>) const;
  std::shared_ptr<RegstDesc> GetProducedRegst4OutEdge(const TaskEdge*) const;

  //
  virtual void ToProto(TaskProto* proto) const {
    ToProto(proto, [](const ChainNode*) { return 0; });
  }
  virtual void ToProto(
      TaskProto*,
      std::function<int64_t(const ChainNode*)> MeaninglessTaskCnt4Chain) const;
  virtual std::string VisualStr() const override;
  virtual TaskType task_type() const = 0;
  std::string DebugStr() const;

 protected:
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const = 0;
  virtual void InitWithFwNode(TaskNode* fw_node);

  ExecGraph& mut_exec_gph() { return exec_gph_; }

  void BindProducedRegstAndOutEdge(std::weak_ptr<RegstDesc>, const TaskEdge*);

  std::shared_ptr<RegstDesc> NewProducedRegstDesc(
      const std::string& regst_desc_name, int32_t min_register_num,
      int32_t max_register_num);

  std::shared_ptr<RegstDesc> NewProducedRegstDesc(
      const std::string& regst_desc_name, int32_t register_num) {
    return NewProducedRegstDesc(regst_desc_name, register_num, register_num);
  }

  void ConsumeRegstDesc(const std::string& regst_desc_name,
                        std::shared_ptr<RegstDesc> regst_desc);

 private:
  // In task_gph level
  const StageNode* stage_node_;
  int64_t thrd_loc_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  int64_t task_id_;
  // In task level
  ExecGraph exec_gph_;

  HashMap<std::string, std::shared_ptr<RegstDesc>> produced_regst_descs_;
  HashMap<std::string, std::weak_ptr<RegstDesc>> consumed_regst_descs_;

  HashMap<std::weak_ptr<RegstDesc>, const TaskEdge*,
          std::function<size_t(const std::weak_ptr<RegstDesc>&)>>
      produced_regst2out_edge_;
  HashMap<const TaskEdge*, std::weak_ptr<RegstDesc>> out_edge2produced_regst_;
};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() { related_fwbp_edge_ = nullptr; }
  ~TaskEdge() = default;

  TaskEdge* related_fwbp_edge() const { return related_fwbp_edge_; }
  void set_related_fwbp_edge(TaskEdge* new_val) {
    related_fwbp_edge_ = new_val;
  }

 private:
  TaskEdge* related_fwbp_edge_;
};

inline std::shared_ptr<RegstDesc> GetRelatedRegst(const TaskEdge* edge) {
  return edge->src_node()->GetProducedRegst4OutEdge(edge);
}

inline const TaskEdge* GetRelatedTaskEdge(std::weak_ptr<RegstDesc> regst) {
  return regst.lock()->GetProducer()->GetOutEdge4ProducedRegst(regst);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_NODE_H_
