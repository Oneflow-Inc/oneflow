#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"
#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/loss_record_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/graph/source_compute_task_node.h"

namespace oneflow {

std::shared_ptr<const Operator> ChainNode::SoleOp() const {
  CHECK_EQ(op_vec_.size(), 1);
  return op_vec_.front();
}

const std::vector<std::shared_ptr<const Operator>>& ChainNode::op_vec() const {
  return op_vec_;
}

std::shared_ptr<const ParallelDesc> ChainNode::parallel_desc() const {
  return parallel_desc_;
}
std::shared_ptr<const ParallelDesc>& ChainNode::mut_parallel_desc() {
  return parallel_desc_;
}

std::string ChainNode::VisualStr() const {
  std::stringstream ss;
  ss << TypeName();
  for (auto op : op_vec_) { ss << "\\n" << op->op_name(); }
  return ss.str();
}

bool ChainNode::HasOpWithModelOrModelTmpBlob() const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (!op->model_bns().empty() || !op->model_tmp_bns().empty()) {
      return true;
    }
  }
  return false;
}

void ChainNode::GenSortedCompTaskNodes(CompTaskNodeHandler Handler) const {
  int64_t parallel_idx = 0;
  int64_t parallel_num = parallel_desc_->parallel_num();
  for (int64_t machine_id : parallel_desc_->sorted_machine_ids()) {
    for (int64_t thrd_id : parallel_desc_->sorted_thrd_loc_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode();
      comp_task_node->set_machine_id(machine_id);
      comp_task_node->set_thrd_loc_id(thrd_id);
      comp_task_node->SetTaskId();
      comp_task_node->set_chain_node(this);
      comp_task_node->mut_parallel_ctx().set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx().set_parallel_num(parallel_num);
      comp_task_node->mut_parallel_ctx().set_policy(parallel_desc_->policy());
      Handler(comp_task_node);
    }
  }
}

CompTaskNode* ForwardChainNode::NewCompTaskNode() const {
  return new FwCompTaskNode;
}
CompTaskNode* BackwardChainNode::NewCompTaskNode() const {
  return new BwCompTaskNode;
}
CompTaskNode* SourceChainNode::NewCompTaskNode() const {
  return new SourceCompTaskNode;
}
CompTaskNode* LossChainNode::NewCompTaskNode() const {
  return new LossCompTaskNode;
}
CompTaskNode* LossAccChainNode::NewCompTaskNode() const {
  return new LossAccCompTaskNode;
}
CompTaskNode* LossRecordChainNode::NewCompTaskNode() const {
  return new LossRecordCompTaskNode;
}
CompTaskNode* MdUpdtChainNode::NewCompTaskNode() const {
  return new MdUpdtCompTaskNode;
}
CompTaskNode* MdSaveChainNode::NewCompTaskNode() const {
  return new MdSaveCompTaskNode;
}
CompTaskNode* MdDiffAccChainNode::NewCompTaskNode() const {
  return new MdDiffAccCompTaskNode;
}

std::string ChainEdge::VisualStr() const { return ""; }
}  // namespace oneflow
