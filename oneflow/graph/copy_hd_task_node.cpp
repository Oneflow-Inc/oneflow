#include "graph/copy_hd_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

const std::vector<std::string>& CopyHDTaskNode::CopiedLbns() const {
  return IsFwInCopy() ? chain_node()->input_lbns() : chain_node()->output_lbns();
}

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

void CopyHDTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

void CopyHDTaskNode::FwBuildExecAndProducedRegsts(Path* path) {
  BindOutEdgeAndRegst();
  // Construct Op
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(IsH2D() ? CopyOpConf::H2D :
                                                             CopyOpConf::D2H);
  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Set ExecNode
  ExecNode* copy_node = mut_exec_gph().NewFinalNode();
  copy_node->mut_op() = copy_op;
  for (const std::string& lbn : CopiedLbns()) {
    copy_node->AddProducedLbnRegstPair(lbn, GetRelatedRegst(SoleOutEdge()));
  }
  // 
  mut_exec_gph().UpdateSourceAndSink();
  AddInPathLbn2ProducedRegst();
}

void CopyHDTaskNode::BpBuildExecAndProducedRegsts(Path* path) {
  BindOutEdgeAndRegst();
  // Get Fw Copy Node
  const ExecGraph& fw_gph = GetFwNode()->exec_gph();
  const ExecNode* fw_copy_node = fw_gph.source_node().SoleOutEdge()->dst_node();
  // Set Bp Copy Node
  ExecNode* bp_copy_node = mut_exec_gph().NewFinalNode();
  bp_copy_node->mut_op() = fw_copy_node->op();
  for (const std::string& lbn : CopiedLbns()) {
    bp_copy_node->AddProducedLbnRegstPair(lbn,
                                         GetRelatedRegst(SoleOutEdge()));
  }
  // 
  mut_exec_gph().UpdateSourceAndSink();
  AddInPathLbn2ProducedRegst();
}

void CopyHDTaskNode::BindOutEdgeAndRegst() {
  std::unique_ptr<RegstDesc> regst_desc(new DisContigRegstDesc);
  BindProducedRegstAndOutEdge(regst_desc.get(), SoleOutEdge());
  AddProducedRegstDesc("cp_out", std::move(regst_desc));
}

} // namespace oneflow
