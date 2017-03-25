#ifndef ONEFLOW_GRAPH_EXEC_GRAPH_H_
#define ONEFLOW_GRAPH_EXEC_GRAPH_H_

#include "operator/operator.h"
#include "graph/graph.h"
#include "graph/register_desc.h"

namespace oneflow {

class TaskNode;
class TaskEdge;
class ExecNode;

class ExecEdge final : public Edge<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecEdge);
  ExecEdge() = default;
  ~ExecEdge() = default;

  const std::string& lbn() const { return lbn_; }
  std::string& mut_lbn() { return lbn_; }

  std::string pbn() const {
    return lbn2pbn(lbn_);
  }

 private:
  std::string lbn2pbn(const std::string& lbn) const {
    return "edge_id_" + std::to_string(edge_id()) + "/" + lbn;
  }

  std::string lbn_;

};

class ExecNode final : public Node<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecNode);
  ExecNode() = default;
  ~ExecNode() = default;

  std::shared_ptr<const Operator> op() const {
    return op_;
  }
  std::shared_ptr<const Operator>& mut_op() {
    return op_;
  }

  const std::vector<std::pair<std::string, TaskEdge*>>& in_task_edges() const {
    return in_task_edges_;
  }
  std::vector<std::pair<std::string, TaskEdge*>>& mut_in_task_edges() {
    return in_task_edges_;
  }

  const std::vector<std::pair<std::string, TaskEdge*>>& out_task_edges() const {
    return out_task_edges_;
  }
  std::vector<std::pair<std::string, TaskEdge*>>& mut_out_task_edges() {
    return out_task_edges_;
  }

  std::string lbn2pbn(const std::string& lbn) const {
    return "node_id_" + std::to_string(node_id()) + "/" + lbn;
  }

 private:
  std::shared_ptr<const Operator> op_;
  std::vector<std::pair<std::string, TaskEdge*>> in_task_edges_;
  std::vector<std::pair<std::string, TaskEdge*>> out_task_edges_;

};

class ExecGraph : public Graph<ExecNode, ExecEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExecGraph);
  ExecGraph() = default;
  virtual ~ExecGraph() = default;

  void set_task_node(TaskNode* task_node) { task_node_ = task_node; }
  void BuildGraph();
  virtual void SetupProducedRegisterDesc() = 0;
  void SubscribeRegisterDescInnerPath();

 protected:
  virtual void FwBuildGraph() = 0;
  virtual void BpBuildGraph() = 0;
  
  const TaskNode* task_node() { return task_node_; }
  void AddProducedRegisterDesc(
      const std::string& register_desc_name,
      std::unique_ptr<RegisterDesc> register_desc) {
    auto pair = std::make_pair(register_desc_name, std::move(register_desc));
    CHECK(produced_register_descs_.insert(std::move(pair)).second);
  }
  RegisterDesc* GetProducedRegisterDesc(const std::string& register_desc_name) {
    return produced_register_descs_.at(register_desc_name).get();
  }

 private:
  TaskNode* task_node_;
  std::unordered_map<std::string, std::unique_ptr<RegisterDesc>> produced_register_descs_; 
  std::unordered_set<RegisterDesc*> subscribed_register_descs_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_EXEC_GRAPH_H_
