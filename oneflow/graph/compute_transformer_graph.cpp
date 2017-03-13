#include "graph/compute_transformer_graph.h"

namespace oneflow {

void CompTransfmGraph::FwBuildFromUserOps() {
  std::unordered_map<std::string, CompTransfmNode*> lbn2producer;
  for (std::shared_ptr<const Operator> op : of_dynamic_cast<const CompTaskNode*>(task_node())->op_vec()) {
    auto cur_node = of_dynamic_cast<CompTransfmNode*>(NewTransfmNode());
    cur_node->mutable_op() = op;
    for (const std::string& obn : op->data_blob_name_set().output_blob_names) {
      std::string lbn = op->obn2lbn(obn);
      lbn2producer[lbn] = cur_node;
      produced_lbn2blob_desc_[lbn].reset(new BlobDescriptor);
      produced_lbn2blob_desc_.at(lbn)->Init();
      produced_lbn2blob_desc_.at(lbn)->mutable_lbn() = lbn;
    }
  }
  for (const std::unique_ptr<Node>& base_node : node_vec()) {
    auto cur_node = of_dynamic_cast<CompTransfmNode*>(base_node.get());
    for (auto ibn : cur_node->op()->data_blob_name_set().input_blob_names) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer.find(lbn);
      if (producer_node_it != lbn2producer.end()) {
        Connect(producer_node_it->second,
                NewTransfmEdge(produced_lbn2blob_desc_.at(lbn).get()),
                cur_node);
      } else {
        extern_in_lbn2consumers_[lbn].push_back(cur_node);
      }
    }
  }
}

void CompTransfmGraph::FwAddCopyInOp() {
  // If only DataOp
  if (extern_in_lbn2consumers_.empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyInOpType());
  pb_op_conf.mutable_copy_op_conf()->clear_logical_blob_names();
  for (const auto& pair : extern_in_lbn2consumers_) {
    pb_op_conf.mutable_copy_op_conf()->add_logical_blob_names(pair.first);
  }
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Transformer Node
  auto copy_node = of_dynamic_cast<CompTransfmNode*>(NewTransfmNode());
  copy_node->mutable_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const auto& pair : extern_in_lbn2consumers_) {
    const std::string& lbn = pair.first;
    const std::vector<CompTransfmNode*>& old_consumers = pair.second;
    produced_lbn2blob_desc_[lbn].reset(new BlobDescriptor);
    produced_lbn2blob_desc_[lbn]->Init();
    BlobDescriptor* blob_desc_ptr = produced_lbn2blob_desc_.at(lbn).get();
    blob_desc_ptr->mutable_lbn() = lbn;
    for (CompTransfmNode* old_consumer : old_consumers) {
      Connect(copy_node, NewTransfmEdge(blob_desc_ptr), old_consumer);
    }
    extern_in_lbn2consumers_.at(lbn) = {copy_node};
  }
}

void CompTransfmGraph::FwAddCloneOp() {
  // collect nodes
  std::vector<CompTransfmNode*> nodes_need_clone;
  for (const std::unique_ptr<Node>& base_node : node_vec()) {
    auto cur_node = of_dynamic_cast<CompTransfmNode*>(base_node.get());
    if (cur_node->op()->data_blob_name_set().output_blob_names.size() <
        cur_node->out_edges().size()) {
      nodes_need_clone.push_back(cur_node);
    }
  }
  // find edges
  for (CompTransfmNode* cur_node : nodes_need_clone) {
    std::unordered_map<BlobDescriptor*, std::vector<CompTransfmEdge*>> blob_desc_ptr2edges;
    for (Edge* base_edge : cur_node->out_edges()) {
      auto edge = of_dynamic_cast<CompTransfmEdge*> (base_edge);
      blob_desc_ptr2edges[edge->blob_desc_ptr()].push_back(edge);
    }
    for (const auto& pair : blob_desc_ptr2edges) {
      if (pair.second.size() <= 1) { continue; }
      BlobDescriptor* blob_desc_ptr = pair.first;
      const std::vector<CompTransfmEdge*>& edges = pair.second;
      // Construct clone op
      OperatorConf pb_op_conf;
      pb_op_conf.set_name("");
      pb_op_conf.mutable_clone_op_conf()->mutable_logical_blob_name()->assign(blob_desc_ptr->lbn());
      pb_op_conf.mutable_clone_op_conf()->set_clone_num(edges.size());
      std::shared_ptr<const Operator> clone_op = ConstructOpFromPbConf(pb_op_conf);
      // Construct Transformer Node
      auto clone_node = of_dynamic_cast<CompTransfmNode*>(NewTransfmNode());
      clone_node->mutable_op() = clone_op;
      // Update Edge
      Connect(cur_node, NewTransfmEdge(blob_desc_ptr), clone_node);
      for (CompTransfmEdge* edge : edges) {
        Node* dst_node = edge->dst_node();
        DisConnect(edge);
        Connect(clone_node, edge, dst_node);
      }
    }
  }
}

} // namespace oneflow
