#ifndef ONEFLOW_GRAPH_TASK_NODE_H_
#define ONEFLOW_GRAPH_TASK_NODE_H_

#include "graph/stage_graph.h"
#include "graph/exec_graph.h"
#include "graph/register_desc.h"

namespace oneflow {

class Path;
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
  const ThrdLocId& thread_local_id() const { return thread_local_id_; }
  const ExecGraph& exec_gph() const { return exec_gph_; }
  
  // Setters
  void SetFwNode() { is_fw_node_ = true; }
  void set_stage_node(const StageNode*);
  ThrdLocId& mut_thread_local_id();

  // return bp_node
  std::unique_ptr<TaskNode> BuildAndConnectBpNode();
  
  //
  void BuildExecAndProducedRegistersAndSubscribeInPath(Path* path);
  void Subscribe(RegiDesc* regi) {
    regi->AddSubscriber(this);
    CHECK(subscribed_regi_descs_.insert(regi).second);
  }
  RegiDesc* GetProducedRegiDesc(const std::string& regi_desc_name) {
    return produced_regi_descs_.at(regi_desc_name).get();
  }

  // 
  const TaskEdge* GetOutEdge4ProducedRegister(RegiDesc*) const;
  RegiDesc* GetProducedRegister4OutEdge(const TaskEdge*) const;
 
 protected:
  virtual std::unique_ptr<TaskNode> CreateSameTypeNode() const;
  virtual void InitWithFwNode(TaskNode* fw_node);

  ExecGraph& mut_exec_gph() { return exec_gph_; }
  
  void BindProducedRegisterAndOutEdge(RegiDesc*, const TaskEdge*);

  void AddProducedRegiDesc(
      const std::string& regi_desc_name,
      std::unique_ptr<RegiDesc> regi_desc) {
    regi_desc->SetProducer(this);
    auto pair = std::make_pair(regi_desc_name, std::move(regi_desc));
    CHECK(produced_regi_descs_.insert(std::move(pair)).second);
  }

  virtual void FwBuildExecAndProducedRegisters(Path*) { UNEXPECTED_RUN(); }
  virtual void BpBuildExecAndProducedRegisters(Path*) { UNEXPECTED_RUN(); }
  void SubscribeRegiDescInnerPath();
  void AddInPathLbn2ProducedRegister();

 private:
  // In task_gph level
  const StageNode* stage_node_;
  ThrdLocId thread_local_id_;
  bool is_fw_node_;
  TaskNode* related_fw_or_bp_node_;
  // In task level
  ExecGraph exec_gph_;
  HashMap<std::string,
                     std::unique_ptr<RegiDesc>> produced_regi_descs_; 
  std::unordered_set<const RegiDesc*> subscribed_regi_descs_;
  HashMap<RegiDesc*, const TaskEdge*> produced_register2out_edge;
  HashMap<const TaskEdge*, RegiDesc*> out_edge2produced_register;

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

inline RegiDesc* GetRelatedRegister(const TaskEdge* edge) {
  return edge->src_node()->GetProducedRegister4OutEdge(edge);
}

inline const TaskEdge* GetRelatedTaskEdge(RegiDesc* regi) {
  return regi->GetProducer()->GetOutEdge4ProducedRegister(regi);
}

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TASK_NODE_H_
