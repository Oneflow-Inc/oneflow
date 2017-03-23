#ifndef ONEFLOW_GRAPH_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_TRANSFM_GRAPH_H_

#include "graph/task_graph.h"
#include "operator/operator.h"

namespace oneflow {

class TransfmNode;

class TransfmEdge final : public Edge<TransfmNode, TransfmEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransfmEdge);
  TransfmEdge() = default;
  ~TransfmEdge() = default;

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

class TransfmNode final : public Node<TransfmNode, TransfmEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransfmNode);
  TransfmNode() = default;
  ~TransfmNode() = default;

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

class TransfmGraph : public Graph<TransfmNode, TransfmEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransfmGraph);
  TransfmGraph() = default;
  virtual ~TransfmGraph() = default;

  void SetTask(TaskNode* task_node) {
    task_node_ = task_node;
  }
  void BuildGraph() {
    if (task_node_->IsFwNode()) {
      FwBuildGraph();
    } else {
      BpBuildGraph();
    }
  }
  virtual void SetProducedRegisterDesc() = 0;

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

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TRANSFM_GRAPH_H_
