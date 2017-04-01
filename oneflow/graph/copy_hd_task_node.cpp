#include "graph/copy_hd_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

const std::vector<std::string>& CopyHDTaskNode::CopiedLbns() const {
  if (IsFwInCopy()) {
    return chain_node()->input_lbns();
  } else {
    return chain_node()->output_lbns();
  }
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
  OperatorConf pb_op_conf;
  pb_op_conf.set_name("");
  pb_op_conf.mutable_copy_op_conf()->set_copy_type(IsH2D() ? CopyOpConf::H2D : CopyOpConf::D2H);

  std::shared_ptr<const Operator> copy_op = ConstructOpFromPbConf(pb_op_conf);

  ExecNode* copy_node = mut_exec_graph().NewExecNode();
  copy_node->mut_op() = copy_op;
  mut_exec_graph().UpdateSourceAndSink();

  std::unique_ptr<RegisterDesc> data_register(new DisContigRegistDesc);
  SoleOutEdge()->set_register_desc(data_register.get());
  AddProducedRegisterDesc("copy", std::move(data_register));
  const std::vector<std::string>& lbns
          = IsFwInCopy() ? chain_node()->input_lbns() :  chain_node()->output_lbns();
  for (const std::string& lbn : lbns) {
    copy_node->AddProducedLbnRegiPair(lbn, SoleOutEdge()->register_desc());
    SoleOutEdge()->register_desc()->AddLbn(lbn);
  }
}

void CopyHDTaskNode::BpBuildExecGraphAndSetProducedRegisterDescs() {
  const ExecGraph& fw_graph = GetFwNode()->exec_graph();
  const ExecNode* cp_in_node = fw_graph.source_node().SoleOutEdge()->dst_node();

  ExecNode* copy_node = mut_exec_graph().NewExecNode();
  copy_node->mut_op() = cp_in_node->op();
  mut_exec_graph().UpdateSourceAndSink();

  std::unique_ptr<RegisterDesc> data_register(new DisContigRegistDesc);
  SoleOutEdge()->set_register_desc(data_register.get());
  AddProducedRegisterDesc("copy", std::move(data_register));
  const std::vector<std::string>& lbns
          = IsFwInCopy() ? chain_node()->input_lbns() :  chain_node()->output_lbns();
  for (const std::string& lbn : lbns) {
    copy_node->AddProducedLbnRegiPair(lbn, SoleOutEdge()->register_desc());
    SoleOutEdge()->register_desc()->AddLbn(lbn);
  }
}

} // namespace oneflow
