#ifndef ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_

#include "graph/task_graph.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class TransfmEdge final : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmEdge);
  TransfmEdge() = default;
  ~TransfmEdge() = default;

  virtual void Init() {
    Edge::Init();
    task_edge_ = nullptr;
  }
 
  const std::string& lbn() const { return lbn_; }
  std::string& mutable_lbn() { return lbn_; }

  const TaskEdge* task_edge() { return task_edge_; }
  void set_task_edge(const TaskEdge* new_task_edge) {
    task_edge_ = new_task_edge;
  }

 private:
  std::string lbn_;
  // It is used when it is a dangling edge
  const TaskEdge* task_edge_;

};

class TransfmNode final : public Node {
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

  bool IsEmptyIn() const override {
    LOG(FATAL) << "TODO";
    return true;
  }
  bool IsEmptyOut() const override {
    LOG(FATAL) << "TODO";
    return true;
  }

 private:
  std::shared_ptr<const Operator> op_;

};

class TransfmGraph : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmGraph);
  TransfmGraph() = default;
  virtual ~TransfmGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) {
    task_node_ = task_node;
    job_has_bp_ = job_has_bp;
    dangling_in_edge_src_.Init();
    dangling_out_edge_dst_.Init();
  }

  virtual void FwBuildGraph() = 0;

 protected:
  TransfmNode* NewTransfmNode() {
    LOG(FATAL) << "TODO";
    return nullptr;
  }
  TransfmEdge* NewTransfmEdge(const std::string& lbn) {
    TransfmEdge* ret = new TransfmEdge;
    ret->Init();
    ret->mutable_lbn() = lbn;
    RegisterEdge(ret);
    return ret;
  }

  const TaskNode* task_node() { return task_node_; }
  bool job_has_bp() { return job_has_bp_; }

  Node* dangling_in_edge_src() {
    return &dangling_in_edge_src_;
  }
  Node* dangling_out_edge_dst() {
    return &dangling_out_edge_dst_;
  }

 private:
  const TaskNode* task_node_;
  bool job_has_bp_;
  
  Node dangling_in_edge_src_;
  Node dangling_out_edge_dst_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
