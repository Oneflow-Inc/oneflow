#include "graph/comp_transfm_graph.h"
#include "graph/register_desc.h"
#include "operator/clone_op.h"
#include "graph/task_node.h"

namespace oneflow {

void CompTransfmGraph::FwBuildGraph() {
  CHECK(task_node()->IsFwNode());
  Lbn2NodeMap lbn2producer;
  Lbn2NodeVecMap extern_in_lbn2consumers;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumers);
  if (task_node()->GetBpNode() != nullptr) {
    FwAddCopyInOp(&extern_in_lbn2consumers);
  }
  FwAddCloneOp();
  FwSetRelatedTaskEdges(lbn2producer, extern_in_lbn2consumers);
  UpdateSourceAndSink();
}

void CompTransfmGraph::FwBuildFromUserOps(
    Lbn2NodeMap* lbn2producer,
    Lbn2NodeVecMap* extern_in_lbn2consumers) {
  for (std::shared_ptr<const Operator> op : task_node()->chain_node()->op_vec()) {
    TransfmNode* cur_node = NewFinalNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_blob_names()) {
      std::string lbn = op->obn2lbn(obn);
      (*lbn2producer)[lbn] = cur_node;
    }
  }
  for (const std::unique_ptr<TransfmNode>& cur_node : nodes()) {
    for (const std::string& ibn : cur_node->op()->input_blob_names()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer->find(lbn);
      if (producer_node_it != lbn2producer->end()) {
        Connect(producer_node_it->second,
                NewTransfmEdge(lbn),
                cur_node.get());
      } else {
        (*extern_in_lbn2consumers)[lbn].push_back(cur_node.get());
      }
    }
  }
}

void CompTransfmGraph::FwAddCopyInOp(Lbn2NodeVecMap* extern_in_lbn2consumers) {
  // If only DataOp
  if (extern_in_lbn2consumers->empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyInOpType());
  pb_op_conf.mutable_copy_op_conf()->clear_lbns();
  for (const auto& pair : *extern_in_lbn2consumers) {
    pb_op_conf.mutable_copy_op_conf()->add_lbns(pair.first);
  }
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Transformer Node
  TransfmNode* copy_node = NewFinalNode();
  copy_node->mut_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const auto& pair : *extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    const std::vector<TransfmNode*>& old_consumers = pair.second;
    for (TransfmNode* old_consumer : old_consumers) {
      Connect(copy_node, NewTransfmEdge(lbn), old_consumer);
    }
    extern_in_lbn2consumers->at(lbn) = {copy_node};
  }
}

void CompTransfmGraph::FwAddCloneOp() {
  struct CloneInfo {
    std::string lbn;
    std::shared_ptr<const Operator> clone_op;
    TransfmNode* pred_node;
    std::vector<TransfmEdge*> edges;
  };
  std::vector<CloneInfo> clone_info_vec;
  // collect clone_info_vec
  for (const std::unique_ptr<TransfmNode>& cur_node : nodes()) {
    std::unordered_map<std::string, std::vector<TransfmEdge*>> lbn2edges;
    for (TransfmEdge* edge : cur_node->out_edges()) {
      lbn2edges[edge->lbn()].push_back(edge);
    }
    for (auto& pair : lbn2edges) {
      if (pair.second.size() <= 1) { continue; }
      const std::string& lbn = pair.first;
      std::vector<TransfmEdge*> edges = pair.second;
      // Construct clone op
      OperatorConf pb_op_conf;
      pb_op_conf.set_name("");
      pb_op_conf.mutable_clone_op_conf()->mutable_lbn()->assign(lbn);
      pb_op_conf.mutable_clone_op_conf()->set_clone_num(edges.size());
      std::shared_ptr<const Operator> clone_op = ConstructOpFromPbConf(pb_op_conf);
      // Set clone_info
      CloneInfo clone_info;
      clone_info.lbn = lbn;
      clone_info.clone_op = clone_op;
      clone_info.pred_node = cur_node.get();
      clone_info.edges = std::move(edges);
      clone_info_vec.push_back(clone_info);
    }
  }
  // Add clone node
  for (const CloneInfo& clone_info : clone_info_vec) {
    TransfmNode* clone_node = NewFinalNode();
    clone_node->mut_op() = clone_info.clone_op;
    // Update Edge
    Connect(clone_info.pred_node, NewTransfmEdge(clone_info.lbn), clone_node);
    for (TransfmEdge* edge : clone_info.edges) {
      TransfmNode* dst_node = edge->dst_node();
      DisConnect(edge);
      Connect(clone_node, edge, dst_node);
    }
  }
}

void CompTransfmGraph::FwSetRelatedTaskEdges(
    const Lbn2NodeMap& lbn2producer,
    const Lbn2NodeVecMap& extern_in_lbn2consumers) {
  // In Task Edge
  for (const auto& pair : extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    for (TransfmNode* consumer : pair.second) {
      consumer->mut_in_task_edges().emplace_back(lbn, task_node()->SoleInEdge());
    }
  }
  // Out Task Edge
  for (const auto& lbn : task_node()->chain_node()->output_lbns()) {
    TransfmNode* producer = lbn2producer.at(lbn);
    producer->mut_out_task_edges().emplace_back(lbn, task_node()->SoleOutEdge());
  }
}

void CompTransfmGraph::BpBuildGraph() {
  const TransfmGraph* fw_graph = task_node()->GetFwNode()->transfm_graph();
  const TransfmNode* cp_in_node = fw_graph->source_node().SoleOutEdge()->dst_node();
  std::unordered_map<const TransfmNode*, TransfmNode*> fw_node2bp_node;
  // Copy Nodes
  for (const std::unique_ptr<TransfmNode>& fw_node : fw_graph->nodes()) {
    if (fw_node.get() == cp_in_node) { continue; }
    TransfmNode* bp_node = NewFinalNode();
    bp_node->mut_op() = fw_node->op();
    fw_node2bp_node.emplace(fw_node.get(), bp_node);
  }
  // Copy Edges
  for (const std::unique_ptr<TransfmEdge>& fw_edge : fw_graph->edges()) {
    if (fw_edge->src_node() == cp_in_node) { continue; }
    Connect(fw_node2bp_node.at(fw_edge->dst_node()),
            NewTransfmEdge(fw_edge->lbn()),
            fw_node2bp_node.at(fw_edge->src_node()));
  }
  // Set in TaskEdge
  for (const std::unique_ptr<TransfmNode>& bp_node : nodes()) {
    if (typeid(*(bp_node->op())) == typeid(CloneOp)) { continue; }
    std::unordered_set<std::string> found_lbns;
    for (TransfmEdge* edge : bp_node->in_edges()) {
      found_lbns.insert(edge->lbn());
    }
    for (const auto& odbn : bp_node->op()->output_diff_blob_names()) {
      std::string lbn = bp_node->op()->odbn2lbn(odbn);
      if (found_lbns.find(lbn) == found_lbns.end()) {
        bp_node->mut_in_task_edges().emplace_back(lbn, task_node()->SoleInEdge());
      }
    }
  }
  // Set out TaskEdge
  for (TransfmEdge* edge : cp_in_node->out_edges()) {
    const std::string& lbn = edge->lbn();
    TransfmNode* bp_node = fw_node2bp_node.at(edge->dst_node());
    bp_node->mut_out_task_edges().emplace_back(lbn, task_node()->SoleOutEdge());
  }
}

void CompTransfmGraph::SetupProducedRegisterDesc() {
  if (task_node()->IsFwNode()) {
    FwSetupProducedRegisterDesc();
  } else {
    BpSetupProducedRegisterDesc();
  }
}

void CompTransfmGraph::FwSetupProducedRegisterDesc() {
  std::unique_ptr<RegisterDesc> data_register(new DisContigRegistDesc);
  data_register->Init();
  // blobs not used by succ_task
  for (const std::unique_ptr<TransfmEdge>& cur_edge : edges()) {
    data_register->Add(cur_edge->pbn());
  }
  for (const std::unique_ptr<TransfmNode>& cur_node : nodes()) {
    for (const std::string& dtbn : cur_node->op()->data_tmp_blob_names()) {
      std::string lbn = cur_node->op()->dtbn2lbn(dtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      data_register->Add(pbn);
    }
  }
  // blobs used by succ_task
  for (const std::unique_ptr<TransfmNode>& cur_node : nodes()) {
    for (auto& pair : cur_node->out_task_edges()) {
      std::string lbn = pair.first;
      std::string pbn = cur_node->lbn2pbn(lbn);
      data_register->Add(pbn, lbn);
    }
  }
  //
  task_node()->SoleOutEdge()->set_register_desc(data_register.get());
  AddProducedRegisterDesc("data", std::move(data_register));
}

void CompTransfmGraph::BpSetupProducedRegisterDesc() {
  std::unique_ptr<RegisterDesc> data_diff_register(new DisContigRegistDesc);
  std::unique_ptr<RegisterDesc> model_diff_register(new ContigRegistDesc);
  std::unique_ptr<RegisterDesc> model_tmp_register(new DisContigRegistDesc);
  data_diff_register->Init();
  model_diff_register->Init();
  model_tmp_register->Init();
  for (const std::unique_ptr<TransfmEdge>& cur_edge : edges()) {
    data_diff_register->Add(cur_edge->pbn());
  }
  for (const std::unique_ptr<TransfmNode>& cur_node : nodes()) {
    for (auto& pair : cur_node->out_task_edges()) {
      std::string lbn = pair.first;
      std::string pbn = cur_node->lbn2pbn(lbn);
      data_diff_register->Add(pbn, lbn);
    }
    for (const std::string& mdbn : cur_node->op()->model_diff_blob_names()) {
      std::string lbn = cur_node->op()->mdbn2lbn(mdbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      model_diff_register->Add(pbn, lbn);
    }
    for (const std::string& mtbn : cur_node->op()->model_tmp_blob_names()) {
      std::string lbn = cur_node->op()->mtbn2lbn(mtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      model_tmp_register->Add(pbn, lbn);
    }
  }
  task_node()->SoleOutEdge()->set_register_desc(data_diff_register.get());
  AddProducedRegisterDesc("data_diff", std::move(data_diff_register));
  AddProducedRegisterDesc("model_diff", std::move(model_diff_register));
  AddProducedRegisterDesc("model_tmp", std::move(model_tmp_register));
}

} // namespace oneflow
