#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void MdSaveCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void MdSaveCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("model", SoleInEdge()->GetSoleRegst());
}

void MdSaveCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnInOpAndRegst(ibn, SoleInEdge()->GetSoleRegst());
  }
}

void MdSaveCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
