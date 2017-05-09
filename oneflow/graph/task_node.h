#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "task/task.pb.h"
#include "graph/stage_graph.h"
#include "graph/exec_graph.h"
#include "register/register_desc_manager.h"

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
  const ChainNode* chain_node() const { return stage_node_->chain_node();}
  const StageNode* stage_node() const { return stage_node_; }
  const uint64_t& thrd_loc_id() const { return thrd_loc_id_; }
  std::string thrd_loc_id_str() const { return std::to_string(thrd_loc_id_); }
  const ExecGraph& exec_gph() const { return exec_gph_; }
  uint64_t task_id() const { return task_id_; }
  std::string task_id_str() const { return std::to_string(task_id_); }
  
  // Setters
  void SetFwNode() { is_fw_node_ = true; }
  void set_stage_node(const StageNode*);
  uint64_t& mut_thrd_loc_id();
  void set_task_id();

  // return bp_node
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();
  
  //
  virtual void BuildExecAndEnrollLbn2Regsts(TaskGraph*) = 0;
  virtual void InferShapeOfBlobsInProducedRegsts(TaskGraph*) = 0;

  #define OVERRIDE_IF_FW_BP_FOR_FUNC(func_name) \
  void func_name(TaskGraph* gph) override { \
    if (IsFwNode()) { \
      return Fw##func_name (gph); \
    } else { \
      return Bp##func_name (gph); \
    } \
  }
  
  //
  RegstDesc* GetProducedRegstDesc(const std::string& regst_desc_name);
  void TakeOverRegstDesc(TaskNode* rhs, const std::string& regst_desc_name);
  const RegstDesc* ForwardedRegstDesc(const std::string& regst_desc_name) const;

  // 
  const TaskEdge* GetOutEdge4ProducedRegst(RegstDesc*) const;
  RegstDesc* GetProducedRegst4OutEdge(const TaskEdge*) const;

  //
  virtual TaskProto ToProto() const;
  virtual std::string VisualStr() const override;
  std::string DebugStr() const;
  
 protected:
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const = 0;
  virtual void InitWithFwNode(TaskNode* fw_node);

  ExecGraph& mut_exec_gph() { return exec_gph_; }
  
  void BindProducedRegstAndOutEdge(RegstDesc*, const TaskEdge*);

  void EnrollProducedRegstDesc(const std::string& regst_desc_name,
                               std::unique_ptr<RegstDesc>&& regst_desc);

 private:
  // In task_gph level
  const StageNode* stage_node_;
  uint64_t thrd_loc_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  uint64_t task_id_;
  // In task level
  ExecGraph exec_gph_;

  HashMap<std::string, std::unique_ptr<RegstDesc>> produced_regst_descs_;
  HashMap<std::string, RegstDesc*> subscribed_regst_descs_;
  HashMap<std::string, const RegstDesc*> forwarded_regst_descs_;

  HashMap<RegstDesc*, const TaskEdge*> produced_regst2out_edge_;
  HashMap<const TaskEdge*, RegstDesc*> out_edge2produced_regst_;

};

class TaskEdge final : public Edge<TaskNode, TaskEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TaskEdge);
  TaskEdge() {
    related_fwbp_edge_ = nullptr;
  }
  ~TaskEdge() = default;

  TaskEdge* related_fwbp_edge() const {
    return related_fwbp_edge_;
  }
  void set_related_fwbp_edge(TaskEdge* new_val) {
    related_fwbp_edge_ = new_val;
  }

 private:
  TaskEdge* related_fwbp_edge_;

};

inline RegstDesc* GetRelatedRegst(const TaskEdge* edge) {
  return edge->src_node()->GetProducedRegst4OutEdge(edge);
}

inline const TaskEdge* GetRelatedTaskEdge(RegstDesc* regst) {
  return regst->GetProducer()->GetOutEdge4ProducedRegst(regst);
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
