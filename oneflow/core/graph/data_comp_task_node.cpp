#include "oneflow/core/graph/data_comp_task_node.h"

namespace oneflow {

void DataCompTaskNode::FwBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  Lbn2NodeBnMap lbn2producer;
  Lbn2NodeBnMap extern_in_lbn2consumer;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumer);
  mut_exec_gph().UpdateSourceAndSink();
  // Enroll Produced Regsts
  if (!out_edges().empty()) {
    auto out_regst = NewProducedRegstDesc("out");
    BindProducedRegstAndOutEdge(out_regst, SoleOutEdge());
  }
  NewProducedRegstDesc("activation");
  NewProducedRegstDesc("data_tmp");
  NewProducedRegstDesc("model_tmp");
  NewProducedRegstDesc("model");
  NewProducedRegstDesc("log");
  // Enroll Lbn
  FwSetExecNodeFromInRegst(extern_in_lbn2consumer);
  FwEnrollLbn2OutRegst(lbn2producer);
  FwEnrollLbn2ActivationRegst();
  FwEnrollLbn2ModelAndTmpRegsts(); // model model_tmp data_tmp
}

void DataCompTaskNode::FwInferShapeOfBlobsInProducedRegsts(TaskGraph*) {
  exec_gph().ConstTopoForEachNode([this](const ExecNode* node) {
    node->op()->InferShape4FwBlobs(
        node->GetMutShapePtr4BnInOpFunc(),
        chain_node()->parallel_desc()->policy(),
        parallel_id(),
        chain_node()->parallel_desc()->parallel_num());
  });
}

void DataCompTaskNode::FwBuildFromUserOps(
    Lbn2NodeBnMap* lbn2producer,
    Lbn2NodeBnMap* extern_in_lbn2consumer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      std::string lbn = op->Lbn4BnInOp(obn);
      CHECK(lbn2producer->insert({lbn, {cur_node, obn}}).second);
    }
  }
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      std::string lbn = cur_node->op()->Lbn4BnInOp(ibn);
      auto producer_it = lbn2producer->find(lbn);
      if (producer_it != lbn2producer->end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        CHECK(extern_in_lbn2consumer->insert({lbn,
                                              {cur_node, ibn}}).second);
      }
    }
  });
}

void DataCompTaskNode::FwSetExecNodeFromInRegst(
    const Lbn2NodeBnMap& extern_in_lbn2consumer) {
  if (extern_in_lbn2consumer.empty()) { return; }
  std::shared_ptr<RegstDesc> in_regst = GetRelatedRegst(SoleInEdge());
  SubscribeRegstDesc("in", in_regst);
  for (const auto& pair : extern_in_lbn2consumer) {
    ExecNode* node = pair.second.first;
    const std::string& ibn = pair.second.second;
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
}

void DataCompTaskNode::FwEnrollLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer) {
  if (IsLossNode()) {
    FwEnrollLbn2OutRegstWhenLoss();
  } else {
    FwEnrollLbn2OutRegstWhenNotLoss(lbn2producer);
  }
}

void DataCompTaskNode::FwEnrollLbn2OutRegstWhenLoss() {
  ExecNode* exec_node = exec_gph().SoleNode();
  // log regst
  std::shared_ptr<RegstDesc> log_regst = GetProducedRegstDesc("log");
  for (const std::string& obn : exec_node->op()->output_bns()) {
    log_regst->EnrollLbn(exec_node->op()->Lbn4BnInOp(obn));
    exec_node->BindBnInOpAndRegst(obn, log_regst);
  }
  // out regst
  if (!out_edges().empty()) {
    std::shared_ptr<RegstDesc> out_regst = GetRelatedRegst(SoleOutEdge());
    for (const std::string& idbn : exec_node->op()->input_diff_bns()) {
      std::string lbn = exec_node->op()->Lbn4BnInOp(idbn);
      out_regst->EnrollLbn(lbn);
      exec_node->BindBnInOpAndRegst(idbn, out_regst);
    }
  }
}

void DataCompTaskNode::FwEnrollLbn2OutRegstWhenNotLoss(
    const Lbn2NodeBnMap& lbn2producer) {
  if (out_edges().empty()) { return; }
  std::shared_ptr<RegstDesc> out_regst = GetRelatedRegst(SoleOutEdge());
  for (const std::string& lbn : chain_node()->output_lbns()) {
    const std::pair<ExecNode*, std::string>& producer = lbn2producer.at(lbn);
    ExecNode* node = producer.first;
    const std::string& obn = producer.second;
    out_regst->EnrollLbn(lbn);
    node->BindBnInOpAndRegst(obn, out_regst);
  }
}

void DataCompTaskNode::FwEnrollLbn2ActivationRegst() {
  auto activation_regst = GetProducedRegstDesc("activation");
  exec_gph().ConstForEachEdge([&](const ExecEdge* edge) {
    activation_regst->EnrollLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
  });
}

void DataCompTaskNode::FwEnrollLbn2ModelAndTmpRegsts() {
  auto data_tmp_regst = GetProducedRegstDesc("data_tmp");
  auto model_tmp_regst = GetProducedRegstDesc("model_tmp");
  auto model_regst = GetProducedRegstDesc("model");
  SubscribeRegstDesc("model_tmp", model_tmp_regst);
  SubscribeRegstDesc("model", model_regst);
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->Lbn4BnInOp(dtbn);
      data_tmp_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      std::string lbn = node->op()->Lbn4BnInOp(mtbn);
      model_tmp_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      std::string lbn = node->op()->Lbn4BnInOp(mbn);
      model_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
  });
}

void DataCompTaskNode::BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  BpBuildExecGraph();
  // New produced registers
  auto in_diff_regst = NewProducedRegstDesc("in_diff");
  if (!out_edges().empty()) {
    BindProducedRegstAndOutEdge(in_diff_regst, SoleOutEdge());
  }
  NewProducedRegstDesc("model_diff");
  NewProducedRegstDesc("activation_diff");
  // Subscribe
  SubscribeRegstDesc("activation",
                     GetFwNode()->GetProducedRegstDesc("activation"));
  SubscribeRegstDesc("data_tmp",
                     GetFwNode()->GetProducedRegstDesc("data_tmp"));
  SubscribeRegstDesc("model", GetFwNode()->GetSubscribedRegstDesc("model"));
  SubscribeRegstDesc("model_tmp",
                     GetFwNode()->GetSubscribedRegstDesc("model_tmp"));
  SubscribeRegstDesc("in", GetFwNode()->GetSubscribedRegstDesc("in"));
  SubscribeRegstDesc("out_diff", GetRelatedRegst(SoleInEdge()));
  // Enroll Lbn
  BpEnrollLbn2ProducedRegst();
}

void DataCompTaskNode::BpInferShapeOfBlobsInProducedRegsts(TaskGraph*) {
  // in_diff_regst
  auto in_diff_regst = GetProducedRegstDesc("in_diff");
  auto in_regst = GetRelatedRegst(GetFwNode()->SoleInEdge());
  in_diff_regst->CopyShapeFrom(in_regst.get());
  // model_diff_regst
  if (auto md_diff_regst = GetProducedRegstDesc("model_diff")) {
    md_diff_regst->CopyShapeFrom(GetFwNode()->GetSubscribedRegstDesc("model").get());
  }
  // activation_diff_regst
  if (auto acti_diff_regst = GetProducedRegstDesc("activation_diff")) {
    auto acti_regst = GetFwNode()->GetProducedRegstDesc("activation");
    acti_diff_regst->CopyShapeFrom(acti_regst.get());
  }
}

void DataCompTaskNode::BpBuildExecGraph() {
  const ExecGraph& fw_gph = GetFwNode()->exec_gph();
  HashMap<const ExecNode*, ExecNode*> fw_node2bp_node;
  fw_gph.ConstForEachNode([&](const ExecNode* fw_node) {
    ExecNode* bp_node = mut_exec_gph().NewNode();
    bp_node->mut_op() = fw_node->op();
    CHECK(fw_node2bp_node.emplace(fw_node, bp_node).second);
  });
  fw_gph.ConstForEachEdge([&](const ExecEdge* fw_edge) {
    ExecEdge* bp_edge = mut_exec_gph().NewEdge();
    bp_edge->set_lbn(fw_edge->lbn());
    bp_edge->mut_src_bn() = GenDiffBn(fw_edge->dst_bn());
    bp_edge->mut_dst_bn() = GenDiffBn(fw_edge->src_bn());
    Connect(fw_node2bp_node.at(fw_edge->dst_node()),
            bp_edge,
            fw_node2bp_node.at(fw_edge->src_node()));
  });
  mut_exec_gph().UpdateSourceAndSink();
}

void DataCompTaskNode::BpEnrollLbn2ProducedRegst() {
  BpEnrollLbn2ActivationDiffRegst();
  BpSetExecNodeFromOutDiffRegst();
  BpEnrollLbn2InDiffRegst();
  BpEnrollLbn2ModelDiffRegst();
}

void DataCompTaskNode::BpEnrollLbn2ActivationDiffRegst() {
  auto activation_regst = GetFwNode()->GetProducedRegstDesc("activation");
  auto activation_diff_regst = GetProducedRegstDesc("activation_diff");
  activation_diff_regst->CopyLbnFrom(activation_regst.get());
  exec_gph().ConstForEachEdge([&](const ExecEdge* edge) {
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_diff_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_diff_regst);
    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()),     
                                         activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()),
                                         activation_regst);
  });
}

void DataCompTaskNode::BpSetExecNodeFromOutDiffRegst() {
  auto out_diff_regst = GetRelatedRegst(SoleInEdge());
  mut_exec_gph().ForEachNode([&](ExecNode* bp_node) {
    std::unordered_set<std::string> found_bns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_bns.insert(edge->dst_bn());
    }
    for (const std::string& odbn : bp_node->op()->output_diff_bns()) {
      if (found_bns.find(odbn) != found_bns.end()) { continue; }
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
    }
  });
}

void DataCompTaskNode::BpEnrollLbn2InDiffRegst() {
  auto in_regst = GetRelatedRegst(GetFwNode()->SoleInEdge());
  auto in_diff_regst = GetProducedRegstDesc("in_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* bp_node) {
    std::unordered_set<std::string> found_bns;
    for (ExecEdge* edge : bp_node->out_edges()) {
      found_bns.insert(edge->src_bn());
    }
    for (const std::string& idbn : bp_node->op()->input_diff_bns()) {
      if (found_bns.find(idbn) != found_bns.end()) { continue; }
      std::string lbn = bp_node->op()->Lbn4BnInOp(idbn);
      in_diff_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      bp_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  });
}

void DataCompTaskNode::BpEnrollLbn2ModelDiffRegst() {
  auto data_tmp_regst = GetFwNode()->GetProducedRegstDesc("data_tmp");
  auto model_tmp_regst = GetFwNode()->GetProducedRegstDesc("model_tmp");
  auto model_diff_regst = GetProducedRegstDesc("model_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mdbn : node->op()->model_diff_bns()) {
      std::string lbn = node->op()->Lbn4BnInOp(mdbn);
      model_diff_regst->EnrollLbn(lbn);
      node->BindBnInOpAndRegst(mdbn, model_diff_regst);
    }
  });
}

} // namespace oneflow
