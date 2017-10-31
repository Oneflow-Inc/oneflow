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
#include "oneflow/core/graph/task_graph.h"

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
    for (int64_t dev_phy_id : parallel_desc_->sorted_dev_phy_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode();
      comp_task_node->set_machine_id(machine_id);
      comp_task_node->set_thrd_loc_id(dev_phy_id);
      comp_task_node->set_chain_node(this);
      comp_task_node->mut_parallel_ctx().set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx().set_parallel_num(parallel_num);
      comp_task_node->mut_parallel_ctx().set_policy(parallel_desc_->policy());
      Handler(comp_task_node);
    }
  }
}

BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromFw() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromBw() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromSrc() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromLoss() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromLossAcc() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromLossRecord() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromMdUpdt() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromMdSave() const {
  UNEXPECTED_RUN();
  return nullptr;
}
BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFromMdDiffAcc() const {
  UNEXPECTED_RUN();
  return nullptr;
}

BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromFw();
}

BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromFw() const {
  return &TaskGraph::BldSubTskGphByNormalBoxing;
}

BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromSrc() const {
  return &TaskGraph::BldSubTskGphByNormalBoxing;
}

BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromMdUpdt() const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromBw();
}

BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromFw() const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromLoss() const {
  return &TaskGraph::BldSubTskGphByNormalBoxing;
}

BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromMdUpdt() const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

BldSubTskGphMthd SourceChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromSrc();
}

BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromLoss();
}

BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromFw() const {
  return &TaskGraph::BldSubTskGphByNormalBoxing;
}

BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromSrc() const {
  return &TaskGraph::BldSubTskGphByNormalBoxing;
}

BldSubTskGphMthd LossAccChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromLossAcc();
}

BldSubTskGphMthd LossAccChainNode::GetMthdForBldSubTskGphFromLoss() const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

BldSubTskGphMthd LossRecordChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode*) const {
  UNEXPECTED_RUN();
  return nullptr;
}

BldSubTskGphMthd LossRecordChainNode::GetMthdForBldSubTskGphFromLossAcc()
    const {
  return &TaskGraph::BldSubTskGphByAddCloneBoxing;
}

BldSubTskGphMthd MdUpdtChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromMdUpdt();
}

BldSubTskGphMthd MdUpdtChainNode::GetMthdForBldSubTskGphFromMdDiffAcc() const {
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    return &TaskGraph::BldSubTskGphByAddCloneBoxing;
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    UNEXPECTED_RUN();
  }
}

BldSubTskGphMthd MdSaveChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode*) const {
  UNEXPECTED_RUN();
  return nullptr;
}

BldSubTskGphMthd MdSaveChainNode::GetMthdForBldSubTskGphFromMdUpdt() const {
  if (parallel_desc()->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
}

BldSubTskGphMthd MdDiffAccChainNode::GetMthdForBldSubTskGphTo(
    const ChainNode* node) const {
  return node->GetMthdForBldSubTskGphFromMdDiffAcc();
}

BldSubTskGphMthd MdDiffAccChainNode::GetMthdForBldSubTskGphFromBw() const {
  return &TaskGraph::BldSubTskGphByOneToOne;
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

BldSubTskGphMthd ChainEdge::GetMthdForBldSubTskGph() const {
  return src_node()->GetMthdForBldSubTskGphTo(dst_node());
}

}  // namespace oneflow
