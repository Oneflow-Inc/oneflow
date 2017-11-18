#include "oneflow/core/graph/model_save_compute_task_node.h"

namespace oneflow {

void MdSaveCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void MdSaveCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("model", SoleInEdge()->GetSoleRegst());
}

void MdSaveCompTaskNode::Build() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = ChainNode()->SoleOp();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnInOpAndRegst(ibn, SoleInEdge()->GetSoleRegst());
  }
}

void MdSaveCompTaskNode::FixThrdLocId() {
  set_thrd_loc_id(IDMgr::Singleton()->PersistenceThrdLocId());
}

}  // namespace oneflow
