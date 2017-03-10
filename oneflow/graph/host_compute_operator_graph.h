#ifndef ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_
#define ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_

#include "graph/compute_operator_graph.h"

namespace oneflow {

class HostCompOpNode final : public ComputeOpNode {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOpNode);
  HostCompOpNode() = default;
  ~HostCompOpNode() = default;

  void Init() {
    ComputeOpNode::Init();
  }

 private:
};

class HostCompOpEdge final : public ComputeOpEdge {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOpEdge);
  HostCompOpEdge() = default;
  ~HostCompOpEdge() = default;

  void Init() {
    ComputeOpEdge::Init();
  }

 private:
};

class HostCompOperatorGraph final : public ComputeOperatorGraph {
 public:
  DISALLOW_COPY_AND_MOVE(HostCompOperatorGraph);
  HostCompOperatorGraph() = default;
  ~HostCompOperatorGraph() = default;

  void Init() {
    ComputeOperatorGraph::Init();
  }

  void FwBuildOperatorGraph(const TaskNode* task_node) {
    // global map
    std::unordered_map<std::string, BlobDescriptor> lbn2blob_desc;
    std::unordered_map<std::string, HostCompOpNode*> lbn2producer;
    std::shared_ptr<Operator> 
    // Build From UserOp
    for (std::shared_ptr<const Operator> op : task_node->op_vec()) {
      HostCompOpNode* cur_node = NewHostCompOpNode();
      cur_node->mutable_op() = op;
      for (auto obn : op->data_blob_desc_set().output_blob_names()) {
        std::string lbn = op->obn2lbn(obn);
        lbn2producer[lbn] = cur_node;
        lbn2blob_desc[lbn]->Init();
      }
    }
    for (std::unique_ptr<Node>& base_node : node_vec()) {
      auto cur_node = of_dynamic_cast<HostCompOpNode*>(base_node.get());
      for (auto ibn : cur_node->op().data_blob_desc_set().input_blob_names()) {
        std::string lbn = cur_node->op().ibn2lbn(ibn);
        auto producer_node_it = lbn2producer.find(lbn);
        if (producer_node_it != lbn2producer.end()) {
          HostCompOpEdge* new_edge = NewHostCompOpEdge();
          new_edge->set_blob_desc_ptr(&(lbn2blob_desc.at(lbn)));
          Connect(producer_node_it->second, new_edge, cur_node);
        } else {
          // TODO:
          // It is not the blob produced by op in this Task
          // we should find it in pre-task later
        }
      }
    }
    // AddCopyD2D
    
    // AddSplit
    // UpdateStartAndStop();
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_HOST_COMPUTE_OPERATOR_GRAPH_H_
