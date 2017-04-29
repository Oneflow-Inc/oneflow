#include "graph/comp_task_node.h"
#include "graph/task_graph.h"
#include "operator/operator_manager.h"
#include "operator/clone_op.h"

namespace oneflow {

std::string CompTaskNode::VisualStr() const override {
  std::stringstream ss;
  ss << TaskNode::VisualStr() 
     << "Compute" << ":"
     << stage_node()->machine_id_str() << ":"
     << thrd_loc_id_str() << "\\n"
     << chain_node()->VisualStr();
  return ss.str();
}

void CompTaskNode::DataFwBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  Lbn2NodeBnMap lbn2producer;
  Lbn2NodeBnMap extern_in_lbn2consumer;
  FwBuildFromUserOps(&lbn2producer, &extern_in_lbn2consumer);
  mut_exec_gph().UpdateSourceAndSink();
  // produced regsts
  auto out_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  auto activation_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  auto data_tmp_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  auto model_tmp_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  // Bind Out Edge
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  // EnrollProducedRegstDesc
  EnrollProducedRegstDesc("out", std::move(out_regst));
  EnrollProducedRegstDesc("activation", std::move(activation_regst));
  EnrollProducedRegstDesc("data_tmp", std::move(data_tmp_regst));
  EnrollProducedRegstDesc("model_tmp", std::move(model_tmp_regst));
  // Enroll Lbn
  FwSetExecNodeFromInRegst(extern_in_lbn2consumer);
  FwEnrollLbn2OutRegst(lbn2producer);
  FwEnrollLbn2ActivationRegst();
  FwEnrollLbn2TmpRegsts();
}

void CompTaskNode::DataFwInferShape4LbnInProducedRegsts() {
  for (const ExecNode& node : exec_gph()) {
    node.op()->InferShape4ObAndDtbFromIb(node.BnInOp2ShapePtr());
    node.op()->InferShape4ModelTmpBlob(node.BnInOp2ShapePtr(),
                                       chain_node()->parallel_desc()->policy(),
                                       parallel_id());
  }
}

void CompTaskNode::MdUpdtFwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  TODO();
  /*
  if (IsFaker()) {
    CompTaskNode* mccoy = gph->faker2mccoy().at(this);
    RegstDesc* regst = mccoy->GetProducedRegstDesc("model_diff");
    BindProducedRegstAndOutEdge(regst, SoleOutEdge());
    return;
  }
  auto model_regst = of_make_unique<ContigRegstDesc> ();
  EnrollProducedRegstDesc("model", std::move(model_regst));
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  mut_exec_gph().UpdateSourceAndSink();
  // PostProcessing in ModelUpdateTaskGraph will complete the work
  // which should be implemented in this function 
  */
}

void CompTaskNode::MdUpdtFwInferShape4LbnInProducedRegsts(TaskGraph* gph) {
  TODO();
}

void CompTaskNode::MdLoadFwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  TODO();
  /*
  if (IsFaker()) {
    CompTaskNode* update_task = gph->faker2mccoy().at(this);
    ExecNode* exec_node = update_task->exec_gph().SoleNode();
    exec_node->BindBnInOpAndRegst("model_init", GetRelatedRegst(SoleInEdge()));
    return;
  }
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  mut_exec_gph().UpdateSourceAndSink();
  auto model_regst = of_make_unique<ContigRegstDesc> ();
  exec_node->BindBnInOpAndRegst(exec_node->op()->SoleObn(), model_regst.get());
  BindProducedRegstAndOutEdge(model_regst.get(), SoleOutEdge());
  Shape* shape_ptr = model_regst->EnrollLbn(RegstDesc::kAllLbn);
  exec_node->op()->SetShapePtr(exec_node->op()->SoleObn(), shape_ptr);
  exec_node->op()->InferShape4ObAndDtbFromIb();
  EnrollProducedRegstDesc("model_regst", std::move(model_regst));
  */
}

void CompTaskNode::MdLoadFwInferShape4LbnInProducedRegsts(TaskGraph* gph) {
  TODO();
}

void CompTaskNode::MdSaveFwBuildExecAndEnrollLbn2Regsts(TaskGraph* gph) {
  TODO();
  /*
  if (IsFaker()) {
    CompTaskNode* update_task = gph->faker2mccoy().at(this);
    RegstDesc* model_regst = update_task->GetProducedRegstDesc("model");
    BindProducedRegstAndOutEdge(model_regst, SoleOutEdge());
    return;
  }
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  mut_exec_gph().UpdateSourceAndSink();
  const std::string& ibn = exec_node->op()->SoleIbn();
  exec_node->BindBnInOpAndRegst(ibn, GetRelatedRegst(SoleInEdge()));
  */
}

void CompTaskNode::MdSaveFwInferShape4LbnInProducedRegsts(TaskGraph* gph) {
  TODO();
}

void CompTaskNode::FwBuildFromUserOps(
    Lbn2NodeBnMap* lbn2producer,
    Lbn2NodeBnMap* extern_in_lbn2consumer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
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
        ExecEdge* edge = mut_exec_gph().NewEdge();
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

void CompTaskNode::FwSetExecNodeFromInRegst(
    const Lbn2NodeBnMap& extern_in_lbn2consumer) {
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  for (const auto& pair : extern_in_lbn2consumer) {
    const std::string& lbn = pair.first;
    Shape* ptr = in_regst->GetMutShapePtr(lbn);
    ExecNode* node = pair.second.first;
    const std::string& ibn = pair.second.second;
    node->BindBnInOpAndShapePtr(ibn, ptr);
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
}

void CompTaskNode::FwEnrollLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer) {
  RegstDesc* out_regst = GetRelatedRegst(SoleOutEdge());
  for (const std::string& lbn : chain_node()->output_lbns()) {
    const std::pair<ExecNode*, std::string>& producer = lbn2producer.at(lbn);
    ExecNode* node = producer.first;
    const std::string& obn = producer.second;
    Shape* ptr = out_regst->EnrollLbn(lbn);
    node->BindBnInOpAndShapePtr(obn, ptr);
    node->BindBnInOpAndRegst(obn, out_regst);
  }
}

void CompTaskNode::FwEnrollLbn2ActivationRegst() {
  RegstDesc* activation_regst = GetProducedRegstDesc("activation");
  for (const std::unique_ptr<ExecEdge>& edge : exec_gph().edges()) {
    Shape* ptr = activation_regst->EnrollLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    edge->src_node()->BindBnInOpAndShapePtr(edge->src_bn(), ptr);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
    edge->dst_node()->BindBnInOpAndShapePtr(edge->dst_bn(), ptr);
  }
}

void CompTaskNode::FwEnrollLbn2TmpRegsts() {
  RegstDesc* data_tmp_regst = GetProducedRegstDesc("data_tmp");
  RegstDesc* model_tmp_regst = GetProducedRegstDesc("model_tmp");
  for (const std::unique_ptr<ExecNode>& node : exec_gph().nodes()) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->dtbn2lbn(dtbn);
      Shape* ptr = data_tmp_regst->EnrollLbn(lbn);
      node->BindBnInOpAndShapePtr(dtbn, ptr);
      node->BindBnInOpAndRegst(dtbn, out_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      std::string lbn = node->op()->mtbn2lbn(mtbn);
      Shape* ptr = model_tmp_regst->EnrollLbn(lbn);
      node->BindBnInOpAndShapePtr(mtbn, ptr);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
  }
}

void CompTaskNode::BpBuildExecAndEnrollLbn2Regsts(TaskGraph*) {
  const ExecGraph& fw_gph = GetFwNode()->exec_gph();
  HashMap<const ExecNode*, ExecNode*> fw_node2bp_node;
  HashMap<ExecEdge*, const ExecEdge*> bp_edge2fw_edge;
  BpBuildExecGraph(fw_gph, &fw_node2bp_node, &bp_edge2fw_edge);
  // Produced registers
  auto in_diff_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  auto model_diff_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  auto activation_diff_regst = RegstDescMgr::Singleton().CreateRegisterDesc();
  // Bind out edge
  BindProducedRegstAndOutEdge(in_diff_regst.get(), SoleOutEdge());
  // Enroll registers
  EnrollProducedRegstDesc("in_diff", std::move(in_diff_regst));
  EnrollProducedRegstDesc("model_diff", std::move(model_diff_regst));
  EnrollProducedRegstDesc("activation_diff", std::move(activation_diff_regst));
  // Enroll Lbn
  BpEnrollLbn2ProducedRegst(fw_node2bp_node, bp_edge2fw_edge);
}

void CompTaskNode::BpInferShape4LbnInProducedRegsts(TaskGraph*) {
  // in_diff_regst
  RegstDesc* in_diff_regst = GetRelatedRegst(SoleOutEdge());
  RegstDesc* in_regst = GetRelatedRegst(GetFwNode()->SoleInEdge());
  in_diff_regst->CopyShapeFrom(in_regst);
  // model_diff_regst
  RegstDesc* model_diff_regst = GetProducedRegstDesc("model_diff");
  for (const std::unique_ptr<ExecNode>& cur_node : exec_gph().nodes()) {
    cur_node->op()->InferShape4ModelDiffBlob(
        cur_node->BnInOp2ShapePtr(),
        chain_node()->parallel_desc()->policy(),
        parallel_id());
  }
  // activation_diff_regst
  RegstDesc* activation_diff_regst = GetProducedRegstDesc("activation_diff");
  RegstDesc* activation_regst = GetFwNode()->GetProducedRegstDesc("activation");
  activation_diff_regst->CopyShapeFrom(activation_regst);
}

void CompTaskNode::BpBuildExecGraph(
    const ExecGraph& fw_gph,
    HashMap<const ExecNode*, ExecNode*>* fw_node2bp_node,
    HashMap<ExecEdge*, const ExecEdge*>* bp_edge2fw_edge) {
  for (const std::unique_ptr<ExecNode>& fw_node : fw_gph.nodes()) {
    ExecNode* bp_node = mut_exec_gph().NewNode();
    bp_node->mut_op() = fw_node->op();
    CHECK(fw_node2bp_node->emplace(fw_node.get(), bp_node).second);
  }
  for (const std::unique_ptr<ExecEdge>& fw_edge : fw_gph.edges()) {
    ExecEdge* bp_edge = mut_exec_gph().NewEdge();
    bp_edge->set_lbn(fw_edge->lbn());
    bp_edge->mut_src_bn() = GenDiffBn(fw_edge->dst_bn());
    bp_edge->mut_dst_bn() = GenDiffBn(fw_edge->src_bn());
    Connect(fw_node2bp_node->at(fw_edge->dst_node()),
            bp_edge,
            fw_node2bp_node->at(fw_edge->src_node()));
    CHECK(bp_edge2fw_edge->emplace(bp_edge, fw_edge.get()).second);
  }
}

void CompTaskNode::BpEnrollLbn2ProducedRegst(
    const HashMap<const ExecNode*, ExecNode*>& fw_node2bp_node,
    const HashMap<ExecEdge*, const ExecEdge*>& bp_edge2fw_edge) {
  // Regsts
  RegstDesc* in_diff_regst = GetRelatedRegst(SoleOutEdge());
  RegstDesc* out_diff_regst = GetRelatedRegst(SoleInEdge());
  RegstDesc* in_regst = GetRelatedRegst(GetFwNode()->SoleInEdge());
  RegstDesc* activation_regst = GetFwNode()->GetProducedRegstDesc("activation");
  RegstDesc* data_tmp_regst = GetFwNode()->GetProducedRegstDesc("data_tmp");
  RegstDesc* model_tmp_regst = GetFwNode()->GetProducedRegstDesc("model_tmp");
  RegstDesc* activation_diff_regst = GetProducedRegstDesc("activation_diff");
  RegstDesc* model_diff_regst = GetProducedRegstDesc("model_diff");
  // blobs on edge 
  for (const std::unique_ptr<ExecEdge>& edge : exec_gph().edges()) {
    activation_diff_regst->EnrollLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_diff_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_diff_regst);
  }
  // extern out_diff blobs
  for (const std::unique_ptr<ExecNode>& bp_node : exec_gph().nodes()) {
    std::unordered_set<std::string> found_bns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_bns.insert(edge->dst_bn());
    }
    for (const std::string& odbn : bp_node->op()->output_diff_bns()) {
      if (found_bns.find(odbn) != found_bns.end()) { continue; }
      std::string lbn = bp_node->op()->odbn2lbn(odbn);
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
    }
  }
  // extern in_diff blobs
  for (const auto& bp_node : exec_gph().nodes()) {
    std::unordered_set<std::string> found_bns;
    for (ExecEdge* edge : bp_node->out_edges()) {
      found_bns.insert(edge->src_bn());
    }
    for (const std::string& idbn : bp_node->op()->input_diff_bns()) {
      if (found_bns.find(idbn) != found_bns.end()) { continue; }
      std::string lbn = bp_node->op()->idbn2lbn(idbn);
      in_diff_regst->EnrollLbn(lbn);
      bp_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      bp_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  }
  // tmp blobs and model_diff blobs
  for (const std::unique_ptr<ExecNode>& node : exec_gph().nodes()) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      std::string lbn = node->op()->dtbn2lbn(dtbn);
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      std::string lbn = node->op()->mtbn2lbn(mtbn);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mdbn : cur_node->op()->model_diff_bns()) {
      std::string lbn = cur_node->op()->mdbn2lbn(mdbn);
      Shape* ptr = model_diff_regst->EnrollLbn(lbn);
      cur_node->BindBnInOpAndShapePtr(mdbn, ptr);
      cur_node->BindBnInOpAndRegst(mdbn, model_diff_regst);
    }
  }
}

void SortByParallelId(std::vector<CompTaskNode*>* comp_node_vec) {
  std::sort(comp_node_vec->begin(), comp_node_vec->end(), []
      (const CompTaskNode* lhs, const CompTaskNode* rhs) {
    return lhs->parallel_id() < rhs->parallel_id();
  });
}

} // namespace oneflow
