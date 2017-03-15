#include "graph/compute_transformer_graph.h"

namespace oneflow {

void CompTransfmGraph::FwBuildFromUserOps() {
  std::unordered_map<std::string, TransfmNode*> lbn2producer;
  for (std::shared_ptr<const Operator> op : task_node()->stage_node()->chain_node()->op_vec()) {
    TransfmNode* cur_node = NewTransfmNode();
    cur_node->mutable_op() = op;
    for (const std::string& obn : op->data_blob_name_set().output_blob_names) {
      std::string lbn = op->obn2lbn(obn);
      lbn2producer[lbn] = cur_node;
    }
  }
  for (const std::unique_ptr<Node>& base_node : node_vec()) {
    auto cur_node = of_dynamic_cast<TransfmNode*>(base_node.get());
    for (auto ibn : cur_node->op()->data_blob_name_set().input_blob_names) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer.find(lbn);
      if (producer_node_it != lbn2producer.end()) {
        Connect(producer_node_it->second,
                NewTransfmEdge(lbn),
                cur_node);
      } else {
        extern_in_lbn2consumers_[lbn].push_back(cur_node);
      }
    }
  }
}

void CompTransfmGraph::FwAddCopyInOp() {
  // If only DataOp
  if (extern_in_lbn2consumers.empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyInOpType());
  pb_op_conf.mutable_copy_op_conf()->clear_logical_blob_names();
  for (const auto& pair : extern_in_lbn2consumers) {
    pb_op_conf.mutable_copy_op_conf()->add_logical_blob_names(pair.first);
  }
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Transformer Node
  TransfmNode* copy_node = NewTransfmNode();
  copy_node->mutable_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const auto& pair : extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    const std::vector<TransfmNode*>& old_consumers = pair.second;
    for (TransfmNode* old_consumer : old_consumers) {
      Connect(copy_node, NewTransfmEdge(lbn), old_consumer);
    }
    extern_in_lbn2consumers.at(lbn) = {copy_node};
  }
}

void CompTransfmGraph::FwAddCloneOp() {
  // find edges
  for (const std::unique_ptr<Node>& base_node : node_vec()) {
    auto cur_node = of_dynamic_cast<TransfmNode*>(base_node.get());
    std::unordered_map<std::string, std::vector<TransfmEdge*>> lbn2edges;
    for (Edge* base_edge : cur_node->out_edges()) {
      auto edge = of_dynamic_cast<TransfmEdge*> (base_edge);
      lbn2edges[edge->lbn()].push_back(edge);
    }
    for (const auto& pair : lbn2edges) {
      if (pair.second.size() <= 1) { continue; }
      const std::string& lbn = pair.first;
      const std::vector<TransfmEdge*>& edges = pair.second;
      // Construct clone op
      OperatorConf pb_op_conf;
      pb_op_conf.set_name("");
      pb_op_conf.mutable_clone_op_conf()->mutable_logical_blob_name()->assign(lbn);
      pb_op_conf.mutable_clone_op_conf()->set_clone_num(edges.size());
      std::shared_ptr<const Operator> clone_op = ConstructOpFromPbConf(pb_op_conf);
      // Construct Transformer Node
      TransfmNode* clone_node = NewTransfmNode();
      clone_node->mutable_op() = clone_op;
      // Update Edge
      Connect(cur_node, NewTransfmEdge(lbn), clone_node);
      for (TransfmEdge* edge : edges) {
        Node* dst_node = edge->dst_node();
        DisConnect(edge); // TODO: delete edge
        Connect(clone_node, edge, dst_node);
      }
    }
  }
}

void CompTransfmGraph::FwAddDanglingEdge() {
  // Add Dangling InEdge
  CHECK_EQ(task_node()->in_edges().size(), 1);
  for (const auto& pair : extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    for (TransfmNode* consumer : pair.second) {
      TransfmEdge* new_edge = NewTransfmEdge(lbn);
      new_edge->set_task_edge(task_node()->in_edges().front());
      Connect(&dangling_in_edge_src_, new_edge, consumer);
    }
  }
  // Add Dangling OutEdge
  CHECK_EQ(task_node()->out_edges().size(), 1);
  for (const auto& lbn : task_node()->stage_node()->chain_node()->output_lbns()) {
    TransfmNode* producer = lbn2producer.at(lbn);
    TransfmEdge* new_edge = NewTransfmEdge(lbn);
    new_edge->set_task_edge(task_node()->out_edges().front());
    Connect(producer, new_edge, &dangling_out_edge_dst_);
  }
}

} // namespace oneflow
