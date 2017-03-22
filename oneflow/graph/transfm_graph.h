#ifndef ONEFLOW_GRAPH_TRANSFM_GRAPH_H_
#define ONEFLOW_GRAPH_TRANSFM_GRAPH_H_

#include "graph/task_graph.h"
#include "operator/operator.h"

namespace oneflow {

class TransfmNode;

class TransfmEdge final : public Edge<TransfmNode, TransfmEdge> {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmEdge);
  TransfmEdge() = default;
  ~TransfmEdge() = default;

  void Init() {
    Edge::Init();
  }
 
  const std::string& lbn() const { return lbn_; }
  std::string& mutable_lbn() { return lbn_; }

  const std::string& pbn() const { return pbn_; }
  std::string& mutable_pbn() { return pbn_; }

 private:
  std::string lbn_;
  std::string pbn_;

};

class TransfmNode final : public Node<TransfmNode, TransfmEdge> {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmNode);
  TransfmNode() = default;
  ~TransfmNode() = default;

  void Init() {
    Node::Init();
  }
  
  std::shared_ptr<const Operator> op() const {
    return op_;
  }
  std::shared_ptr<const Operator>& mutable_op() {
    return op_;
  }

  const std::vector<std::pair<std::string, TaskEdge*>>& in_task_edges() const {
    return in_task_edges_;
  }
  std::vector<std::pair<std::string, TaskEdge*>>& mutable_in_task_edges() {
    return in_task_edges_;
  }

  const std::vector<std::pair<std::string, TaskEdge*>>& out_task_edges() const {
    return out_task_edges_;
  }
  std::vector<std::pair<std::string, TaskEdge*>>& mutable_out_task_edges() {
    return out_task_edges_;
  }

 private:
  std::shared_ptr<const Operator> op_;
  std::vector<std::pair<std::string, TaskEdge*>> in_task_edges_;
  std::vector<std::pair<std::string, TaskEdge*>> out_task_edges_;

};

class TransfmGraph : public Graph<TransfmNode, TransfmEdge> {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmGraph);
  TransfmGraph() = default;
  virtual ~TransfmGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) {
    task_node_ = task_node;
    job_has_bp_ = job_has_bp;
  }

  virtual void FwBuildGraph() = 0;
  virtual void BpBuildGraph() = 0;
  virtual void SetProducedRegisterDesc() = 0;

 protected:
  const TaskNode* task_node() { return task_node_; }
  bool job_has_bp() { return job_has_bp_; }
  void AddProducedRegisterDesc(const std::string& register_desc_name,
                               std::unique_ptr<RegisterDesc> register_desc_ptr) {
    CHECK(produced_register_descs_.insert(std::make_pair(
            register_desc_name,
            std::move(register_desc_ptr))).second);
  }
  RegisterDesc& GetProducedRegisterDesc(const std::string& register_desc_name) {
    return *(produced_register_descs_.at(register_desc_name));
  }

 private:
  const TaskNode* task_node_;
  bool job_has_bp_;
  std::unordered_map<std::string, std::unique_ptr<RegisterDesc>> produced_register_descs_; 

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TRANSFM_GRAPH_H_
