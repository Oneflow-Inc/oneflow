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

void CopyHDTaskNode::FwBuildExecGraphAndSetProducedRegisterDescs() {
  BindOutEdgeAndRegister();
  // Construct Op
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(IsH2D() ? CopyOpConf::H2D : CopyOpConf::D2H);
  std::shared_ptr<Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);
  // Set ExecNode
  ExecNode* copy_node = mut_exec_graph().NewExecNode();
  copy_node->mut_op() = copy_op;
  for (const std::string& lbn : CopiedLbns()) {
    copy_node->AddProducedLbnRegiPair(lbn, GetRelatedRegister(SoleOutEdge()));
  }
  // 
  mut_exec_graph().UpdateSourceAndSink();
  AddInPathLbn2ProducedRegister();
}

void CopyHDTaskNode::BpBuildExecGraphAndSetProducedRegisterDescs() {
  BindOutEdgeAndRegister();
  // Get Fw Copy Node
  const ExecGraph& fw_graph = GetFwNode()->exec_graph();
  const ExecNode* fw_copy_node = fw_graph.source_node().SoleOutEdge()->dst_node();
  // Set Bp Copy Node
  ExecNode* bp_copy_node = mut_exec_graph().NewExecNode();
  bp_copy_node->mut_op() = fw_copy_node->op();
  for (const std::string& lbn : CopiedLbns()) {
    bp_copy_node->AddProducedLbnRegiPair(lbn, GetRelatedRegister(SoleOutEdge()));
  }
  // 
  mut_exec_graph().UpdateSourceAndSink();
  AddInPathLbn2ProducedRegister();
}

void CopyHDTaskNode::BindOutEdgeAndRegister() {
  std::unique_ptr<RegisterDesc> register_desc(new DisContigRegistDesc);
  BindProducedRegisterAndOutEdge(register_desc.get(), SoleOutEdge());
  AddProducedRegisterDesc("cp_out", std::move(register_desc));
}

} // namespace oneflow
