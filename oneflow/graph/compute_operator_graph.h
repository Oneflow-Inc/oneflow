#ifndef ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_

#include "graph/operator_graph.h"

namespace oneflow {

class ComputeOpNode : OpNode {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOpNode);
  ComputeOpNode() = default;
  virtual ComputeOpNode() = default;

  virtual void Init() {
    OpNode::Init();
    // struct style
  }

 private:
  std::shared_ptr<const Operator> op_;

};

class ComputeOpEdge : OpEdge {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOpEdge);
  ComputeOpEdge() = default;
  virtual ~ComputeOpEdge() = default;

  virtual void Init() {
    OpEdge::Init();
  }

 private:
};

class ComputeOperatorGraph : public OperatorGraph {
 public:
  DISALLOW_COPY_AND_MOVE(ComputeOperatorGraph);
  ComputeOperatorGraph() = default;
  virtual ComputeOperatorGraph() = default;

  virtual void Init(); // TODO

  virtual void FwBuildOperatorGraph(const TaskNode* task_node) {
    std::unordered_map<std::string, ComputeOpNode*> lbn2producer;
    for (std::shared_ptr<const Operator> op : task_node->op_vec()) {
      ComputeOpNode* cur_node = NewComputeOpNode();
      cur_node->mutable_op() = op;
      for (auto obn : op->data_blob_desc_set().output_blob_names()) {
        std::string lbn = op->OBN2LBN(obn);
        lbn2producer[lbn] = cur_node;
        lbn2blob_desc_[lbn]->Init();
      }
    }
    for (std::unique_ptr<Node>& base_node : node_vec()) {
      auto cur_node = of_dynamic_cast<ComputeOpNode*>(base_node.get());
      for (auto ibn : cur_node->op().data_blob_desc_set().input_blob_names()) {
        std::string lbn = cur_node->op().IBN2LBN(ibn);
        auto producer_node_it = lbn2producer.find(lbn);
        if (producer_node_it != lbn2producer.end()) {
          ComputeOpEdge* new_edge = NewComputeOpEdge();
          new_edge->set_blob_desc_ptr(&(lbn2blob_desc_.at(lbn)));
          Connect(producer_node_it->second, new_edge, cur_node);
        } else {
          lbn2blob_desc_[lbn]->Init();
        }
      }
    }
    // TODO: AddCopyD2D
    // TODO: AddSplit
    UpdateStartAndStop();
  }

 private:
  std::unordered_map<std::string, BlobDescriptor> lbn2blob_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_COMPUTE_OPERATOR_GRAPH_H_
