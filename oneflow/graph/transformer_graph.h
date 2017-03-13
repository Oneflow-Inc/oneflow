#ifndef ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_

#include "graph/task_graph.h"
#include "blob/blob_descriptor.h"

namespace oneflow {

class TransfmNode : public Node {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmNode);
  TransfmNode() = default;
  virtual ~TransfmNode() = default;

  virtual void Init() {
    Node::Init();
    // struct style
  }
  
  std::shared_ptr<const Operator> op() const {
    return op_;
  }
  std::shared_ptr<const Operator>& mutable_op() {
    return op_;
  }

 private:
  std::shared_ptr<const Operator> op_;
};

class TransfmEdge : public Edge {
 public:
  DISALLOW_COPY_AND_MOVE(TransfmEdge);
  TransfmEdge() = default;
  virtual ~TransfmEdge() = default;

  virtual void Init() {
    Edge::Init();
    // struct style
  }
  
  BlobDescriptor* blob_desc_ptr() const {
    return blob_desc_ptr_;
  }
  void set_blob_desc_ptr(BlobDescriptor* new_blob_desc_ptr) {
    blob_desc_ptr_ = new_blob_desc_ptr;
  }

 private:
  BlobDescriptor* blob_desc_ptr_;
};

class TransformerGraph : public Graph {
 public:
  DISALLOW_COPY_AND_MOVE(TransformerGraph);
  TransformerGraph() = default;
  virtual ~TransformerGraph() = default;

  virtual void Init(const TaskNode* task_node, bool job_has_bp) {
    task_node_ = task_node;
    job_has_bp_ = job_has_bp;
  }

  virtual void FwBuildGraph() = 0;

 protected:
  virtual TransfmNode* NewTransfmNode() = 0;
  virtual TransfmEdge* NewTransfmEdge(BlobDescriptor*) = 0;

  const TaskNode* task_node() { return task_node_; }
  bool job_has_bp() { return job_has_bp_; }

 private:
  const TaskNode* task_node_;
  bool job_has_bp_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_TRANSFORMER_GRAPH_H_
