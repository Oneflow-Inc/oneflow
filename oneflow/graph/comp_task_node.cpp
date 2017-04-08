#include "graph/comp_task_node.h"
#include "operator/operator_factory.h"
#include "operator/clone_op.h"
#include "path/path.h"

namespace oneflow {

bool CompTaskNode::HasOpWithOutDiff() const {
  for (auto op : chain_node()->op_vec()) {
    if (! op->output_diff_blob_names().empty()) {
      return true;
    }
  }
  return false;
}

bool CompTaskNode::HasOpWithIndiff() const {
  for (auto op : chain_node()->op_vec()) {
    if (! op->input_diff_blob_names().empty()) {
      return true;
    }
  }
  return false;
}

void CompTaskNode::FwBuildExecAndProducedRegisters(Path* path) {
  (this->*(path->MemFunc4FwBuildExecAndProducedRegisters()))(path);
}

void CompTaskNode::DataFwBuildExecAndProducedRegisters(Path* path) {
  Lbn2NodeMap lbn2producer;
  Lbn2NodeVecMap extern_in_lbn2consumers;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumers);
  if (GetBpNode() != nullptr) {
    FwAddCopyInOp(&extern_in_lbn2consumers);
  }
  FwAddCloneOp();
  mut_exec_graph().UpdateSourceAndSink();
  FwBindOutEdgeAndRegister();
  FwSetRegisterPtrs4ExecNodes(lbn2producer, extern_in_lbn2consumers);
  FwSetProducedRegisterDescs();
}

void CompTaskNode::ModelUpdateFwBuildExecAndProducedRegisters(Path* path) {
  if (IsFaker()) {
    CompTaskNode* mccoy = path->faker2mccoy().at(this);
    RegisterDesc* regi = mccoy->GetProducedRegisterDesc("model_diff");
    BindProducedRegisterAndOutEdge(regi, SoleOutEdge());
    return;
  }
  TODO();
}

void CompTaskNode::ModelLoadFwBuildExecAndProducedRegisters(Path*) {
  TODO();
}

void CompTaskNode::ModelSaveFwBuildExecAndProducedRegisters(Path*) {
  TODO();
}

void CompTaskNode::FwBuildFromUserOps(
    Lbn2NodeMap* lbn2producer,
    Lbn2NodeVecMap* extern_in_lbn2consumers) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_graph().NewFinalNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_blob_names()) {
      std::string lbn = op->obn2lbn(obn);
      (*lbn2producer)[lbn] = cur_node;
    }
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_graph().nodes()) {
    for (const std::string& ibn : cur_node->op()->input_blob_names()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer->find(lbn);
      if (producer_node_it != lbn2producer->end()) {
        Connect(producer_node_it->second,
                mut_exec_graph().NewExecEdge(lbn),
                cur_node.get());
      } else {
        (*extern_in_lbn2consumers)[lbn].push_back(cur_node.get());
      }
    }
  }
}

void CompTaskNode::FwAddCopyInOp(Lbn2NodeVecMap* extern_in_lbn2consumers) {
  // If only DataOp
  if (extern_in_lbn2consumers->empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyInOpType());
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Exec Node
  ExecNode* copy_node = mut_exec_graph().NewFinalNode();
  copy_node->mut_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const auto& pair : *extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    const std::vector<ExecNode*>& old_consumers = pair.second;
    for (ExecNode* old_consumer : old_consumers) {
      Connect(copy_node, mut_exec_graph().NewExecEdge(lbn), old_consumer);
    }
    extern_in_lbn2consumers->at(lbn) = {copy_node};
  }
}

void CompTaskNode::FwAddCloneOp() {
  struct CloneInfo {
    std::string lbn;
    std::shared_ptr<const Operator> clone_op;
    ExecNode* pred_node;
    std::vector<ExecEdge*> edges;
  };
  std::vector<CloneInfo> clone_info_vec;
  // collect clone_info_vec
  for (const std::unique_ptr<ExecNode>& cur_node : exec_graph().nodes()) {
    std::unordered_map<std::string, std::vector<ExecEdge*>> lbn2edges;
    for (ExecEdge* edge : cur_node->out_edges()) {
      lbn2edges[edge->lbn()].push_back(edge);
    }
    for (auto& pair : lbn2edges) {
      if (pair.second.size() <= 1) { continue; }
      const std::string& lbn = pair.first;
      std::vector<ExecEdge*>& edges = pair.second;
      // Construct clone op
      OperatorConf pb_op_conf;
      pb_op_conf.set_name("");
      pb_op_conf.mutable_clone_op_conf();
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
    ExecNode* clone_node = mut_exec_graph().NewFinalNode();
    clone_node->mut_op() = clone_info.clone_op;
    // Update Edge
    Connect(clone_info.pred_node, mut_exec_graph().NewExecEdge(clone_info.lbn), clone_node);
    for (ExecEdge* edge : clone_info.edges) {
      ExecNode* dst_node = edge->dst_node();
      DisConnect(edge);
      Connect(clone_node, edge, dst_node);
    }
  }
}

void CompTaskNode::FwBindOutEdgeAndRegister() {
  std::unique_ptr<RegisterDesc> data_register(new DisContigRegistDesc);
  BindProducedRegisterAndOutEdge(data_register.get(), SoleOutEdge());
  AddProducedRegisterDesc("data", std::move(data_register));
}

void CompTaskNode::FwSetRegisterPtrs4ExecNodes(
    const Lbn2NodeMap& lbn2producer,
    const Lbn2NodeVecMap& extern_in_lbn2consumers) {
  // In Register Desc
  for (const auto& pair : extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    for (ExecNode* consumer : pair.second) {
      consumer->AddConsumedLbnRegiPair(lbn, GetRelatedRegister(SoleInEdge()));
    }
  }
  // Out Register Desc
  for (const auto& lbn : chain_node()->output_lbns()) {
    ExecNode* producer = lbn2producer.at(lbn);
    producer->AddProducedLbnRegiPair(lbn, GetRelatedRegister(SoleOutEdge()));
  }
}

void CompTaskNode::FwSetProducedRegisterDescs() {
  RegisterDesc* data_register = GetRelatedRegister(SoleOutEdge());
  for (const std::unique_ptr<ExecEdge>& cur_edge : exec_graph().edges()) {
    data_register->AddPbn(cur_edge->pbn());
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_graph().nodes()) {
    for (const std::string& dtbn : cur_node->op()->data_tmp_blob_names()) {
      std::string lbn = cur_node->op()->dtbn2lbn(dtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      data_register->AddPbn(pbn);
    }
  }
  AddInPathLbn2ProducedRegister();
}

void CompTaskNode::BpBuildExecAndProducedRegisters(Path* path) {
  const ExecGraph& fw_graph = GetFwNode()->exec_graph();
  const ExecNode* cp_in_node = fw_graph.source_node().SoleOutEdge()->dst_node();
  std::unordered_map<const ExecNode*, ExecNode*> fw_node2bp_node;
  BpBuildExecGraph(fw_graph, cp_in_node, &fw_node2bp_node);
  BpBindOutEdgeAndRegister();
  BpSetRegisterDescPtrs4Nodes(cp_in_node, fw_node2bp_node);
  BpSetProducedRegisterDescs();
}

void CompTaskNode::BpBuildExecGraph(
    const ExecGraph& fw_graph,
    const ExecNode* cp_in_node,
    std::unordered_map<const ExecNode*, ExecNode*>* fw_node2bp_node) {
  for (const std::unique_ptr<ExecNode>& fw_node : fw_graph.nodes()) {
    if (fw_node.get() == cp_in_node) { continue; }
    ExecNode* bp_node = mut_exec_graph().NewFinalNode();
    bp_node->mut_op() = fw_node->op();
    fw_node2bp_node->emplace(fw_node.get(), bp_node);
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_graph.edges()) {
    if (fw_edge->src_node() == cp_in_node) { continue; }
    Connect(fw_node2bp_node->at(fw_edge->dst_node()),
            mut_exec_graph().NewExecEdge(fw_edge->lbn()),
            fw_node2bp_node->at(fw_edge->src_node()));
  }
}

void CompTaskNode::BpBindOutEdgeAndRegister() {
  std::unique_ptr<RegisterDesc> data_diff_register(new DisContigRegistDesc);
  BindProducedRegisterAndOutEdge(data_diff_register.get(), SoleOutEdge());
  AddProducedRegisterDesc("data_diff", std::move(data_diff_register));
}

void CompTaskNode::BpSetRegisterDescPtrs4Nodes(
    const ExecNode* cp_in_node,
    const std::unordered_map<const ExecNode*, ExecNode*>& fw_node2bp_node) {
  // Set in register_desc
  for (const std::unique_ptr<ExecNode>& bp_node : exec_graph().nodes()) {
    if (typeid(*(bp_node->op())) == typeid(CloneOp)) { continue; }
    std::unordered_set<std::string> found_lbns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_lbns.insert(edge->lbn());
    }
    for (const auto& odbn : bp_node->op()->output_diff_blob_names()) {
      std::string lbn = bp_node->op()->odbn2lbn(odbn);
      if (found_lbns.find(lbn) == found_lbns.end()) {
        bp_node->AddConsumedLbnRegiPair(lbn, GetRelatedRegister(SoleInEdge()));
      }
    }
  }
  // Set out register_desc
  for (ExecEdge* edge : cp_in_node->out_edges()) {
    const std::string& lbn = edge->lbn();
    ExecNode* bp_node = fw_node2bp_node.at(edge->dst_node());
    bp_node->AddProducedLbnRegiPair(lbn, GetRelatedRegister(SoleOutEdge()));
  }
}

void CompTaskNode::BpSetProducedRegisterDescs() {
  std::unique_ptr<RegisterDesc> model_diff_register(new ContigRegistDesc);
  std::unique_ptr<RegisterDesc> model_tmp_register(new DisContigRegistDesc);
  RegisterDesc* data_diff_register = GetRelatedRegister(SoleOutEdge());
  for (const std::unique_ptr<ExecEdge>& cur_edge : exec_graph().edges()) {
    data_diff_register->AddPbn(cur_edge->pbn());
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_graph().nodes()) {
    for (const std::string& mdbn : cur_node->op()->model_diff_blob_names()) {
      std::string lbn = cur_node->op()->mdbn2lbn(mdbn);
      model_diff_register->AddLbn(lbn);
    }
    for (const std::string& mtbn : cur_node->op()->model_tmp_blob_names()) {
      std::string lbn = cur_node->op()->mtbn2lbn(mtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      model_tmp_register->AddPbn(pbn);
    }
  }
  AddInPathLbn2ProducedRegister();
  AddProducedRegisterDesc("model_diff", std::move(model_diff_register));
  AddProducedRegisterDesc("model_tmp", std::move(model_tmp_register));
}

} // namespace oneflow
