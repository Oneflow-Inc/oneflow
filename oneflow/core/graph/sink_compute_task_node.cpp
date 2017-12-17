#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/model_save_compute_task_node.h"

namespace oneflow {

void SinkCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void SinkCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void SinkCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = chain_node()->SoleOp();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnInOpAndRegst(ibn, SoleInEdge()->GetSoleRegst());
  }
  CHECK(node->op()->data_tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

void SinkCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
