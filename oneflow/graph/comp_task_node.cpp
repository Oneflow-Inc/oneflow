#include "graph/comp_task_node.h"
#include "operator/operator_factory.h"
#include "operator/clone_op.h"
#include "path/path.h"

namespace oneflow {

void CompTaskNode::FwBuildExecAndProducedRegsts(Path* path) {
  (this->*(path->Func4FwBuildExecAndProducedRegsts()))(path);
}

void CompTaskNode::DataFwBuildExecAndProducedRegsts(Path* path) {
  Lbn2NodeMap lbn2producer;
  Lbn2NodeVecMap extern_in_lbn2consumers;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumers);
  if (GetBpNode() != nullptr) {
    FwAddCopyInOp(&extern_in_lbn2consumers);
  }
  FwAddCloneOp();
  mut_exec_gph().UpdateSourceAndSink();
  FwBindOutEdgeAndRegst();
  FwSetRegstPtrs4ExecNodes(lbn2producer, extern_in_lbn2consumers);
  FwSetProducedRegstDescs();
}

void CompTaskNode::ModelUpdateFwBuildExecAndProducedRegsts(Path* path) {
  if (IsFaker()) {
    CompTaskNode* mccoy = path->Faker2Mccoy(this);
    RegstDesc* regst = mccoy->GetProducedRegstDesc("model_diff");
    BindProducedRegstAndOutEdge(regst, SoleOutEdge());
    return;
  }
  std::unique_ptr<RegstDesc> model_regst(new ContigRegstDesc);
  ExecNode* exec_node = mut_exec_gph().NewFinalNode();
  exec_node->mut_op() = chain_node()->op_vec().front();
  mut_exec_gph().UpdateSourceAndSink();
  for (std::shared_ptr<const Operator> op : path->GetDataChain()->op_vec()) {
    for (const std::string& mbn : op->model_bns()) {
      std::string lbn = op->mbn2lbn(mbn);
      exec_node->AddConsumedLbnRegstPair(lbn, GetRelatedRegst(SoleInEdge()));
      exec_node->AddProducedLbnRegstPair(lbn, model_regst.get());
    }
  }
  AddProducedRegstDesc("model", std::move(model_regst));
  AddInPathLbn2ProducedRegst();
}

void CompTaskNode::ModelLoadFwBuildExecAndProducedRegsts(Path*) {
  TODO();
}

void CompTaskNode::ModelSaveFwBuildExecAndProducedRegsts(Path*) {
  TODO();
}

void CompTaskNode::FwBuildFromUserOps(
    Lbn2NodeMap* lbn2producer,
    Lbn2NodeVecMap* extern_in_lbn2consumers) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewFinalNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      std::string lbn = op->obn2lbn(obn);
      (*lbn2producer)[lbn] = cur_node;
    }
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_node_it = lbn2producer->find(lbn);
      if (producer_node_it != lbn2producer->end()) {
        Connect(producer_node_it->second,
                mut_exec_gph().NewExecEdge(lbn),
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
  ExecNode* copy_node = mut_exec_gph().NewFinalNode();
  copy_node->mut_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const auto& pair : *extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    const std::vector<ExecNode*>& old_consumers = pair.second;
    for (ExecNode* old_consumer : old_consumers) {
      Connect(copy_node, mut_exec_gph().NewExecEdge(lbn), old_consumer);
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
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    HashMap<std::string, std::vector<ExecEdge*>> lbn2edges;
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
      pb_op_conf.mutable_clone_op_conf()->set_out_num(edges.size());
      auto clone_op = ConstructOpFromPbConf(pb_op_conf);
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
    ExecNode* clone_node = mut_exec_gph().NewFinalNode();
    clone_node->mut_op() = clone_info.clone_op;
    // Update Edge
    Connect(clone_info.pred_node,
            mut_exec_gph().NewExecEdge(clone_info.lbn),
            clone_node);
    for (ExecEdge* edge : clone_info.edges) {
      ExecNode* dst_node = edge->dst_node();
      DisConnect(edge);
      Connect(clone_node, edge, dst_node);
    }
  }
}

void CompTaskNode::FwBindOutEdgeAndRegst() {
  std::unique_ptr<RegstDesc> data_regst(new DisContigRegstDesc);
  BindProducedRegstAndOutEdge(data_regst.get(), SoleOutEdge());
  AddProducedRegstDesc("data", std::move(data_regst));
}

void CompTaskNode::FwSetRegstPtrs4ExecNodes(
    const Lbn2NodeMap& lbn2producer,
    const Lbn2NodeVecMap& extern_in_lbn2consumers) {
  // In Regst Desc
  for (const auto& pair : extern_in_lbn2consumers) {
    const std::string& lbn = pair.first;
    for (ExecNode* consumer : pair.second) {
      consumer->AddConsumedLbnRegstPair(lbn, GetRelatedRegst(SoleInEdge()));
    }
  }
  // Out Regst Desc
  for (const auto& lbn : chain_node()->output_lbns()) {
    ExecNode* producer = lbn2producer.at(lbn);
    producer->AddProducedLbnRegstPair(lbn, GetRelatedRegst(SoleOutEdge()));
  }
}

void CompTaskNode::FwSetProducedRegstDescs() {
  RegstDesc* data_regst = GetRelatedRegst(SoleOutEdge());
  for (const std::unique_ptr<ExecEdge>& cur_edge : exec_gph().edges()) {
    data_regst->EnrollWithPbnAndLbn(cur_edge->pbn(), cur_edge->lbn());
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    for (const std::string& dtbn : cur_node->op()->data_tmp_bns()) {
      std::string lbn = cur_node->op()->dtbn2lbn(dtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      data_regst->EnrollWithPbnAndLbn(pbn, lbn);
    }
  }
  AddInPathLbn2ProducedRegst();
}

void CompTaskNode::BpBuildExecAndProducedRegsts(Path* path) {
  const ExecGraph& fw_gph = GetFwNode()->exec_gph();
  const ExecNode* cp_in_node = fw_gph.source_node().SoleOutEdge()->dst_node();
  HashMap<const ExecNode*, ExecNode*> fw_node2bp_node;
  BpBuildExecGraph(fw_gph, cp_in_node, &fw_node2bp_node);
  BpBindOutEdgeAndRegst();
  BpSetRegstDescPtrs4Nodes(cp_in_node, fw_node2bp_node);
  BpSetProducedRegstDescs();
}

void CompTaskNode::BpBuildExecGraph(
    const ExecGraph& fw_gph,
    const ExecNode* cp_in_node,
    HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node) {
  for (const std::unique_ptr<ExecNode>& fw_node : fw_gph.nodes()) {
    if (fw_node.get() == cp_in_node) { continue; }
    ExecNode* bp_node = mut_exec_gph().NewFinalNode();
    bp_node->mut_op() = fw_node->op();
    fw_node2bp_node->emplace(fw_node.get(), bp_node);
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_gph.edges()) {
    if (fw_edge->src_node() == cp_in_node) { continue; }
    Connect(fw_node2bp_node->at(fw_edge->dst_node()),
            mut_exec_gph().NewExecEdge(fw_edge->lbn()),
            fw_node2bp_node->at(fw_edge->src_node()));
  }
}

void CompTaskNode::BpBindOutEdgeAndRegst() {
  std::unique_ptr<RegstDesc> data_diff_regst(new DisContigRegstDesc);
  BindProducedRegstAndOutEdge(data_diff_regst.get(), SoleOutEdge());
  AddProducedRegstDesc("data_diff", std::move(data_diff_regst));
}

void CompTaskNode::BpSetRegstDescPtrs4Nodes(
    const ExecNode* cp_in_node,
    const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node) {
  // Set in regst_desc
  for (const std::unique_ptr<ExecNode>& bp_node : exec_gph().nodes()) {
    if (typeid(*(bp_node->op())) == typeid(CloneOp)) { continue; }
    std::unordered_set<std::string> found_lbns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_lbns.insert(edge->lbn());
    }
    for (const auto& obn : bp_node->op()->output_bns()) {
      std::string lbn = bp_node->op()->obn2lbn(obn);
      if (found_lbns.find(lbn) == found_lbns.end()) {
        bp_node->AddConsumedLbnRegstPair(lbn, GetRelatedRegst(SoleInEdge()));
      }
    }
  }
  // Set out regst_desc
  for (ExecEdge* edge : cp_in_node->out_edges()) {
    const std::string& lbn = edge->lbn();
    ExecNode* bp_node = fw_node2bp_node.at(edge->dst_node());
    bp_node->AddProducedLbnRegstPair(lbn, GetRelatedRegst(SoleOutEdge()));
  }
}

void CompTaskNode::BpSetProducedRegstDescs() {
  std::unique_ptr<RegstDesc> model_diff_regst(new ContigRegstDesc);
  std::unique_ptr<RegstDesc> model_tmp_regst(new DisContigRegstDesc);
  RegstDesc* data_diff_regst = GetRelatedRegst(SoleOutEdge());
  for (const std::unique_ptr<ExecEdge>& cur_edge : exec_gph().edges()) {
    data_diff_regst->EnrollWithPbnAndLbn(cur_edge->pbn(), cur_edge->lbn());
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    for (const std::string& mbn : cur_node->op()->model_bns()) {
      std::string lbn = cur_node->op()->mbn2lbn(mbn);
      model_diff_regst->EnrollWithLbn(lbn);
    }
    for (const std::string& mtbn : cur_node->op()->model_tmp_bns()) {
      std::string lbn = cur_node->op()->mtbn2lbn(mtbn);
      std::string pbn = cur_node->lbn2pbn(lbn);
      model_tmp_regst->EnrollWithPbnAndLbn(pbn, lbn);
    }
  }
  AddInPathLbn2ProducedRegst();
  AddProducedRegstDesc("model_diff", std::move(model_diff_regst));
  AddProducedRegstDesc("model_tmp", std::move(model_tmp_regst));
}

} // namespace oneflow
