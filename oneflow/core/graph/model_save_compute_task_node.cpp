#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

void MdSaveCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void MdSaveCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("model", SoleInEdge()->GetSoleRegst());
}

void MdSaveCompTaskNode::Build() {
  OperatorConf op_conf;
  op_conf.set_name("model_save_op");  // TODO
  op_conf.mutable_model_save_conf();
  SoleInEdge()->GetSoleRegst()->ForEachLbn([&](const std::string& lbn) {
    op_conf.mutable_model_save_conf()->add_lbns(lbn);
  });
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = OpMgr::Singleton()->AddOp(op_conf);
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnInOpAndRegst(ibn, SoleInEdge()->GetSoleRegst());
  }
}

void MdSaveCompTaskNode::FixThrdLocId() {
  set_thrd_loc_id(IDMgr::Singleton()->PersistenceThrdLocId());
}

}  // namespace oneflow
