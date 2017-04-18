#include "graph/comp_task_node.h"
#include "graph/task_graph.h"
#include "operator/operator_factory.h"
#include "operator/clone_op.h"

namespace oneflow {

void CompTaskNode::FwBuildExecAndProducedRegsts(TaskGraph* gph) {
  (this->*(gph->Func4FwBuildExecAndProducedRegsts()))(gph);
}

void CompTaskNode::DataFwBuildExecAndProducedRegsts(TaskGraph*) {
  Lbn2NodeBnMap lbn2producer;
  Lbn2NodeBnMap extern_in_lbn2consumer;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumer);
  if (GetBpNode() != nullptr) {
    FwAddCopyInOp(&extern_in_lbn2consumer);
  }
  mut_exec_gph().UpdateSourceAndSink();
  // data regst
  auto data_regst = make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(data_regst.get(), SoleOutEdge());
  EnrollProducedRegstDesc("data", std::move(data_regst));
  FwSetDataRegstDesc(lbn2producer, extern_in_lbn2consumer);
  // model_tmp regst
  auto model_tmp_regst = make_unique<DisContigRegstDesc> ();
  EnrollProducedRegstDesc("model_tmp", std::move(model_tmp_regst));
  FwSetModelTmpRegstDesc();
}

void CompTaskNode::MdUpdtFwBuildExecAndProducedRegsts(TaskGraph* gph) {
  if (IsFaker()) {
    CompTaskNode* mccoy = gph->faker2mccoy().at(this);
    RegstDesc* regst = mccoy->GetProducedRegstDesc("model_diff");
    BindProducedRegstAndOutEdge(regst, SoleOutEdge());
    return;
  }
  auto model_regst = make_unique<ContigRegstDesc> ();
  EnrollProducedRegstDesc("model", std::move(model_regst));
  ExecNode* exec_node = mut_exec_gph().NewFinalNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  mut_exec_gph().UpdateSourceAndSink();
  // PostProcessing in ModelUpdateTaskGraph will complete the work
  // which should be implemented in this function 
}

void CompTaskNode::MdLoadFwBuildExecAndProducedRegsts(TaskGraph* gph) {
  if (IsFaker()) {
    CompTaskNode* update_task = gph->faker2mccoy().at(this);
    ExecNode* exec_node = update_task->exec_gph().SoleNode();
    exec_node->BindBnInOpAndRegst("model_init", GetRelatedRegst(SoleInEdge()));
    return;
  }
  ExecNode* exec_node = mut_exec_gph().NewFinalNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  mut_exec_gph().UpdateSourceAndSink();
  auto model_regst = make_unique<ContigRegstDesc> ();
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleObn(), model_regst.get());
  BindProducedRegstAndOutEdge(model_regst.get(), SoleOutEdge());
  Shape* shape_ptr = model_regst->EnrollLbn(RegstDesc::kAllLbn);
  exec_node->op()->SetShapePtr(exec_node->op()->SoleObn(), shape_ptr);
  exec_node->op()->InferShape4ObAndDtbFromIb();
  EnrollProducedRegstDesc("model_regst", std::move(model_regst));
}

void CompTaskNode::MdSaveFwBuildExecAndProducedRegsts(TaskGraph*) {
  TODO();
}

void CompTaskNode::FwBuildFromUserOps(
    Lbn2NodeBnMap* lbn2producer,
    Lbn2NodeBnMap* extern_in_lbn2consumer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewFinalNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      std::string lbn = op->obn2lbn(obn);
      CHECK(lbn2producer->insert({lbn, {cur_node, obn}}).second);
    }
  }
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      std::string lbn = cur_node->op()->ibn2lbn(ibn);
      auto producer_it = lbn2producer->find(lbn);
      if (producer_it != lbn2producer->end()) {
        ExecEdge* edge = mut_exec_gph().NewFinalEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node.get());
      } else {
        CHECK(extern_in_lbn2consumer->insert({lbn,
                                              {cur_node.get(), ibn}}).second);
      }
    }
  }
}

void CompTaskNode::FwAddCopyInOp(Lbn2NodeBnMap* extern_in_lbn2consumer) {
  if (extern_in_lbn2consumer->empty()) { return; }
  // Construct Copy Operator
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("copy_in_" + std::to_string(node_id()));
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(CopyInOpType());
  for (const auto& pair : *extern_in_lbn2consumer) {
    pb_op_conf.mutable_copy_op_conf()->add_copied_lbns(pair.first);
  }
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Construct Exec Node
  ExecNode* copy_node = mut_exec_gph().NewFinalNode();
  copy_node->mut_op() = copy_op;
  // Connect CopyNode and OldConsumer
  for (const std::string& obn : copy_node->op()->output_bns()) {
    std::string lbn = copy_node->op()->obn2lbn(obn);
    const auto& old_consumer = extern_in_lbn2consumer->at(lbn);
    ExecEdge* edge = mut_exec_gph().NewFinalEdge();
    edge->set_lbn(lbn);
    edge->mut_dst_bn() = old_consumer.second;
    edge->mut_src_bn() = obn;
    Connect(copy_node, edge, old_consumer.first);
  }
  for (const std::string& ibn : copy_node->op()->input_bns()) {
    std::string lbn = copy_node->op()->ibn2lbn(ibn);
    extern_in_lbn2consumer->at(lbn) = {copy_node, ibn};
  }
}

void CompTaskNode::FwSetDataRegstDesc(
    const Lbn2NodeBnMap& lbn2producer,
    const Lbn2NodeBnMap& extern_in_lbn2consumer) {
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* out_regst = GetRelatedRegst(SoleOutEdge());
  // blob on exec_edge
  for (const std::unique_ptr<ExecEdge>& edge : exec_gph().edges()) {
    Shape* ptr = out_regst->EnrollLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), out_regst);
    edge->src_node()->op()->SetShapePtr(edge->src_bn(), ptr);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), out_regst);
    edge->dst_node()->op()->SetShapePtr(edge->dst_bn(), ptr);
  }
  // extern in blobs
  for (const auto& pair : extern_in_lbn2consumer) {
    const std::string& lbn = pair.first;
    Shape* ptr = in_regst->GetMutShapePtr(lbn);
    ExecNode* node = pair.second.first;
    const std::string& ibn = pair.second.second;
    node->op()->SetShapePtr(ibn, ptr);
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
  // extern out blobs
  for (const std::string& lbn : chain_node()->output_lbns()) {
    const std::pair<ExecNode*, std::string>& producer = lbn2producer.at(lbn);
    ExecNode* node = producer.first;
    const std::string& obn = producer.second;
    Shape* ptr = out_regst->EnrollLbn(lbn);
    node->op()->SetShapePtr(obn, ptr);
    node->BindBnInOpAndRegst(obn, out_regst);
  }
  // data tmp blobs
  for (const std::unique_ptr<ExecNode>& node : exec_gph().nodes()) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->dtbn2lbn(dtbn);
      Shape* ptr = out_regst->EnrollLbn(lbn);
      node->op()->SetShapePtr(dtbn, ptr);
      node->BindBnInOpAndRegst(dtbn, out_regst);
    }
  }
  // Inference Shape
  for (const ExecNode& node : exec_gph()) {
    node.op()->InferShape4ObAndDtbFromIb();
  }
}

void CompTaskNode::FwSetModelTmpRegstDesc() {
  RegstDesc* model_tmp_regst = GetProducedRegstDesc("model_tmp");
  for (const std::unique_ptr<ExecNode>& node : exec_gph().nodes()) {
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      std::string lbn = node->op()->mtbn2lbn(mtbn);
      Shape* ptr = model_tmp_regst->EnrollLbn(lbn);
      node->op()->SetShapePtr(mtbn, ptr);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    node->op()->InferShape4Mtb();
  }
}

void CompTaskNode::BpBuildExecAndProducedRegsts(TaskGraph*) {
  const ExecGraph& fw_gph = GetFwNode()->exec_gph();
  const ExecNode* cp_in_node = fw_gph.source_node().SoleOutEdge()->dst_node();
  HashMap<const ExecNode*, ExecNode*> fw_node2bp_node;
  HashMap<ExecEdge*, const ExecEdge*> bp_edge2fw_edge;
  BpBuildExecGraph(fw_gph, cp_in_node, &fw_node2bp_node, &bp_edge2fw_edge);
  //
  auto data_diff_regst = make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(data_diff_regst.get(), SoleOutEdge());
  EnrollProducedRegstDesc("data_diff", std::move(data_diff_regst));
  BpSetDataDiffRegst(cp_in_node, fw_node2bp_node, bp_edge2fw_edge);
  //
  auto model_diff_regst = make_unique<ContigRegstDesc> ();
  EnrollProducedRegstDesc("model_diff", std::move(model_diff_regst));
  BpSetModelDiffRegst();
}

void CompTaskNode::BpBuildExecGraph(
    const ExecGraph& fw_gph,
    const ExecNode* cp_in_node,
    HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node,
    HashMap<ExecEdge*, const ExecEdge*>* bp_edge2fw_edge) {
  for (const std::unique_ptr<ExecNode>& fw_node : fw_gph.nodes()) {
    if (fw_node.get() == cp_in_node) { continue; }
    ExecNode* bp_node = mut_exec_gph().NewFinalNode();
    bp_node->mut_op() = fw_node->op();
    fw_node2bp_node->emplace(fw_node.get(), bp_node);
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_gph.edges()) {
    if (fw_edge->src_node() == cp_in_node) { continue; }
    ExecEdge* bp_edge = mut_exec_gph().NewFinalEdge();
    bp_edge->set_lbn(fw_edge->lbn());
    bp_edge->mut_src_bn() = GenDiffBn(fw_edge->dst_bn());
    bp_edge->mut_dst_bn() = GenDiffBn(fw_edge->src_bn());
    Connect(fw_node2bp_node->at(fw_edge->dst_node()),
            bp_edge,
            fw_node2bp_node->at(fw_edge->src_node()));
    CHECK(bp_edge2fw_edge->emplace(bp_edge, fw_edge.get()).second);
  }
}

void CompTaskNode::BpSetDataDiffRegst(
    const ExecNode* cp_in_node,
    const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node,
    const HashMap<ExecEdge*, const ExecEdge*>& bp_edge2fw_edge) {
  // Regsts
  RegstDesc* in_diff_regst = GetRelatedRegst(SoleOutEdge());
  RegstDesc* out_diff_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* out_regst = GetRelatedRegst(GetFwNode()->SoleOutEdge());
  RegstDesc* model_tmp_regst = GetFwNode()->GetProducedRegstDesc("model_tmp");
  // blobs on edge 
  for (const std::unique_ptr<ExecEdge>& edge : exec_gph().edges()) {
    Shape* ptr = in_diff_regst->EnrollLbn(edge->lbn());
    *ptr = out_regst->GetShape(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), in_diff_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), in_diff_regst);
  }
  // extern out_diff blobs
  for (const std::unique_ptr<ExecNode>& bp_node : exec_gph().nodes()) {
    std::unordered_set<std::string> found_bns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_bns.insert(edge->dst_bn());
    }
    for (const auto& odbn : bp_node->op()->output_diff_bns()) {
      if (found_bns.find(odbn) != found_bns.end()) { continue; }
      std::string lbn = bp_node->op()->odbn2lbn(odbn);
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
    }
  }
  // extern in_diff blobs
  for (ExecEdge* edge : cp_in_node->out_edges()) {
    ExecNode* bp_node = fw_node2bp_node.at(edge->dst_node());
    Shape* ptr = in_diff_regst->EnrollLbn(edge->lbn());
    *ptr = out_regst->GetShape(edge->lbn());
    bp_node->BindBnInOpAndRegst(GenDiffBn(edge->dst_bn()), in_diff_regst);
  }
  // tmp blobs
  for (const std::unique_ptr<ExecNode>& node : exec_gph().nodes()) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->dtbn2lbn(dtbn);
      node->BindBnInOpAndRegst(dtbn, out_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      std::string lbn = node->op()->mtbn2lbn(mtbn);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
  }
}

void CompTaskNode::BpSetModelDiffRegst() {
  RegstDesc* model_diff_regst = GetProducedRegstDesc("model_diff");
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    for (const std::string& mdbn : cur_node->op()->model_diff_bns()) {
      std::string lbn = cur_node->op()->mdbn2lbn(mdbn);
      Shape* ptr = model_diff_regst->EnrollLbn(lbn);
      cur_node->op()->SetShapePtr(mdbn, ptr);
      cur_node->BindBnInOpAndRegst(mdbn, model_diff_regst);
    }
    cur_node->op()->InferShape4Mdb();
  }
}

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec) {
  std::sort(comp_node_vec->begin(), comp_node_vec->end(), []
      (const CompTaskNode* lhs, const CompTaskNode* rhs) {
    return lhs->parallel_id() < rhs->parallel_id();
  });
}

} // namespace oneflow
