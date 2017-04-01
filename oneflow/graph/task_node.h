#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "graph/stage_graph.h"
#include "graph/exec_graph.h"
#include "graph/register_desc.h"

namespace oneflow {

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
  const ThreadLocalId& thread_local_id() const { return thread_local_id_; }
  const ExecGraph& exec_graph() const { return exec_graph_; }
  
  // Setters
  void SetFwNode() { is_fw_node_ = true; }
  void set_stage_node(const StageNode*);
  ThreadLocalId& mut_thread_local_id();

  //
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();
  void BuildExecGraphAndSetRegisterDescs();

  // 
  const TaskEdge* GetOutEdge4ProducedRegister(RegisterDesc*) const;
  RegisterDesc* GetProducedRegister4OutEdge(const TaskEdge*) const;
 
 protected:
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const;
  virtual void InitWithFwNode(TaskNode* fw_node);

  ExecGraph& mut_exec_graph() { return exec_graph_; }
  
  void BindProducedRegisterAndOutEdge(RegisterDesc*, const TaskEdge*);

  void AddProducedRegisterDesc(
      const std::string& register_desc_name,
      std::unique_ptr<RegisterDesc> register_desc) {
    register_desc->SetProducer(this);
    auto pair = std::make_pair(register_desc_name, std::move(register_desc));
    CHECK(produced_register_descs_.insert(std::move(pair)).second);
  }
  RegisterDesc* GetProducedRegisterDesc(const std::string& register_desc_name) {
    return produced_register_descs_.at(register_desc_name).get();
  }

  virtual void FwBuildExecGraphAndSetProducedRegisterDescs() { UNEXPECTED_RUN(); }
  virtual void BpBuildExecGraphAndSetProducedRegisterDescs() { UNEXPECTED_RUN(); }
  void SubscribeRegisterDescInnerPath();
  void AddInPathLbn2ProducedRegister();

 private:
  // In task_graph level
  const StageNode* stage_node_;
  ThreadLocalId thread_local_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  // In task level
  ExecGraph exec_graph_;
  std::unordered_map<std::string,
                     std::unique_ptr<RegisterDesc>> produced_register_descs_; 
  std::unordered_set<const RegisterDesc*> subscribed_register_descs_;
  std::unordered_map<RegisterDesc*, const TaskEdge*> produced_register2out_edge;
  std::unordered_map<const TaskEdge*, RegisterDesc*> out_edge2produced_register;

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

inline RegisterDesc* GetRelatedRegister(const TaskEdge* edge) {
  return edge->src_node()->GetProducedRegister4OutEdge(edge);
}

inline const TaskEdge* GetRelatedTaskEdge(RegisterDesc* regi) {
  return regi->GetProducer()->GetOutEdge4ProducedRegister(regi);
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
