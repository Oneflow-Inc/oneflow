#include "graph/copy_hd_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

void CopyHDTaskNode::BuildExecAndProducedRegstsForCopy(TaskGraph* gph){
  OperatorConf op_conf;
  op_conf.set_name("copy_" + NewUniqueId());
  CopyOpConf* copy_conf = op_conf.mutable_copy_conf();
  copy_conf->set_copy_type(
      IsH2D() ? CopyOpConf::H2D : CopyOpConf::D2H);
  for(std::string lbn : CopiedLbns()){
    copy_conf->add_copied_lbns(lbn);
  }
  RegstDesc* in_regst = GetRelatedRegst(SoleInEdge());
  if(copy_conf->copied_lbns_size() == 1 
      && copy_conf->copied_lbns(0) == RegstDesc::kAllLbn){
    copy_conf->clear_copied_lbns();
    for(auto& pair : in_regst->GetLbn2ShapeMap()){
      copy_conf->add_copied_lbns(pair.first);
    }
  }
  ExecNode* node = mut_exec_gph().NewFinalNode();
  node->mut_op() = ConstructOpFromPbConf(op_conf);

  auto out_regst = of_make_unique<DisContigRegstDesc> ();
  BindProducedRegstAndOutEdge(out_regst.get(), SoleOutEdge());
  EnrollProducedRegstDesc("copy", std::move(out_regst));

  for(std::string ibn : node->op()->input_bns()){
    std::string lbn = node->op()->ibn2lbn(ibn);
    Shape* shape_ptr = in_regst->GetMutShapePtr(lbn);
    node->op()->SetShapePtr(ibn, shape_ptr);
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
  for(std::string obn : node->op()->output_bns()){
    std::string lbn = node->op()->obn2lbn(obn);
    Shape* shape_ptr = out_regst->EnrollLbn(lbn);
    node->op()->SetShapePtr(obn, shape_ptr);
    node->BindBnInOpAndRegst(obn, out_regst.get());
  }
  node->op()->InferShape4ObAndDtbFromIb();
  mut_exec_gph().UpdateSourceAndSink();
}

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

const std::vector<std::string>& CopyHDTaskNode::CopiedLbns() const {
  return IsFwInCopy() ? chain_node()->input_lbns() : chain_node()->output_lbns();
}

void CopyHDTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

void CopyHDTaskNode::FwBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForCopy(gph);
}

void CopyHDTaskNode::BpBuildExecAndProducedRegsts(TaskGraph* gph) {
  BuildExecAndProducedRegstsForCopy(gph);
}

} // namespace oneflow
