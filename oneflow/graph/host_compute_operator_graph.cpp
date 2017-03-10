#include "graph/host_compute_operator_graph.h"
#include "operator/operator_factory.h"

namespace oneflow {

void HostCompOperatorGraph::BuildFromUserOps() {
  std::unordered_map<std::string, HostCompOpNode*> lbn2producer;
  for (std::shared_ptr<const Operator> op : task_node_->op_vec()) {
    HostCompOpNode* cur_node = NewHostCompOpNode();
    cur_node->mutable_op() = op;
    for (auto obn : op->data_blob_name_set().output_blob_names) {
      std::string lbn = op->obn2lbn(obn);
      lbn2producer[lbn] = cur_node;
      produced_lbn2blob_desc_[lbn].Init();
    }
  }
  for (const std::unique_ptr<Node>& base_node : node_vec()) {
    auto cur_node = of_dynamic_cast<HostCompOpNode*>(base_node.get());
    for (auto ibn : cur_node->op()->data_blob_name_set().input_blob_names) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer.find(lbn);
      if (producer_node_it != lbn2producer.end()) {
        HostCompOpEdge* new_edge = NewHostCompOpEdge();
        new_edge->set_blob_desc_ptr(&(produced_lbn2blob_desc_.at(lbn)));
        Connect(producer_node_it->second, new_edge, cur_node);
      } else {
        input_lbn2consumer_[lbn] = cur_node;
      }
    }
  }
}

void HostCompOperatorGraph::AddCopyInOp() {
  // Only DataOp
  if (input_lbn2consumer_.empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyOpConf::H2H);
  pb_op_conf.mutable_copy_op_conf()->clear_logical_blob_names();
  for (const auto& pair : input_lbn2consumer_) {
    pb_op_conf.mutable_copy_op_conf()->add_logical_blob_names(pair.first);
  }
  std::shared_ptr<const Operator> copy_op = OperatorFactory::singleton().ConstructOp(pb_op_conf);
  // Construct Copy Node
  HostCompOpNode* copy_node = NewHostCompOpNode();
  copy_node->mutable_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (auto& pair : input_lbn2consumer_) {
    const std::string& lbn = pair.first;
    HostCompOpNode* old_consumer = pair.second;
    produced_lbn2blob_desc_[lbn].Init();
    HostCompOpEdge* new_edge = NewHostCompOpEdge();
    new_edge->set_blob_desc_ptr(&(produced_lbn2blob_desc_.at(lbn)));
    Connect(copy_node, new_edge, old_consumer);
    input_lbn2consumer_.at(lbn) = copy_node;
  }
}

void HostCompOperatorGraph::AddCloneOp() {
  
}

} // namespace oneflow
