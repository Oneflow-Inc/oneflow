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

namespace {

BldBoxingOpMthd GetBldBoxingOpMethodByFwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  TODO();
}
BldBoxingOpMthd GetBldBoxingOpMethodByBwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  TODO();
}
std::vector<std::string> FindLbnsBetweenFw(const ChainNode* in_chain,
                                           const ChainNode* out_chain) {
  TODO();
}
std::vector<std::string> FindLbnsBetweenBw(const ChainNode* in_chain,
                                           const ChainNode* out_chain) {
  TODO();
}

}  // namespace

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

#define DEFINE_VIRTUAL_METHOD(x)                                               \
  const char* x##ChainNode::TypeName() const { return #x "ChainNode"; }        \
  BldSubTskGphMthd x##ChainNode::GetMthdForBldSubTskGphTo(                     \
      const ChainNode* node) const {                                           \
    return node->GetMthdForBldSubTskGphFrom##x(this);                          \
  }                                                                            \
  BldBoxingOpMthd x##ChainNode::GetMthdForBldBoxingOpTo(const ChainNode* node) \
      const {                                                                  \
    return node->GetMthdForBldBoxingOpFrom##x(this);                           \
  }                                                                            \
  std::vector<std::string> x##ChainNode::FindLbnsTo(const ChainNode* node)     \
      const {                                                                  \
    return node->FindLbnsFrom##x(this);                                        \
  }                                                                            \
  BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFrom##x(const ChainNode*)  \
      const {                                                                  \
    UNEXPECTED_RUN();                                                          \
    return nullptr;                                                            \
  }                                                                            \
  BldBoxingOpMthd ChainNode::GetMthdForBldBoxingOpFrom##x(const ChainNode*)    \
      const {                                                                  \
    UNEXPECTED_RUN();                                                          \
    return nullptr;                                                            \
  }                                                                            \
  std::vector<std::string> ChainNode::FindLbnsFrom##x(const ChainNode*)        \
      const {                                                                  \
    UNEXPECTED_RUN();                                                          \
    return {};                                                                 \
  }                                                                            \
  CompTaskNode* x##ChainNode::NewCompTaskNode() const {                        \
    return new x##CompTaskNode;                                                \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, CHAIN_TYPE_SEQ)

// ForwardChainNode
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromSource(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpMthd ForwardChainNode::GetMthdForBldBoxingOpFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByFwParallelPolicy(node, this);
}
BldBoxingOpMthd ForwardChainNode::GetMthdForBldBoxingOpFromSource(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromSource(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}

// BackwardChainNode
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromBackward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpMthd BackwardChainNode::GetMthdForBldBoxingOpFromBackward(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByBwParallelPolicy(node, this);
}
BldBoxingOpMthd BackwardChainNode::GetMthdForBldBoxingOpFromLoss(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByBwParallelPolicy(node, this);
}
std::vector<std::string> BackwardChainNode::FindLbnsFromBackward(
    const ChainNode* node) const {
  return FindLbnsBetweenBw(node, this);
}
std::vector<std::string> BackwardChainNode::FindLbnsFromLoss(
    const ChainNode* node) const {
  return FindLbnsBetweenBw(node, this);
}

// LossChainNode
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromSource(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpMthd LossChainNode::GetMthdForBldBoxingOpFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByFwParallelPolicy(node, this);
}
BldBoxingOpMthd LossChainNode::GetMthdForBldBoxingOpFromSource(
    const ChainNode* node) const {
  return GetBldBoxingOpMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> LossChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> LossChainNode::FindLbnsFromSource(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}

// LossAccChainNode
BldSubTskGphMthd LossAccChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

// LossRecordChainNode
BldSubTskGphMthd LossRecordChainNode::GetMthdForBldSubTskGphFromLossAcc(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpMthd LossRecordChainNode::GetMthdForBldBoxingOpFromLossAcc(
    const ChainNode*) const {
  return &BoxingTaskNode::BldBoxingOpWithAddClone;
}
std::vector<std::string> LossRecordChainNode::FindLbnsFromLossAcc(
    const ChainNode*) const {
  return {kPackedBlobName};
}

// MdUpdtChainNode
BldSubTskGphMthd MdUpdtChainNode::GetMthdForBldSubTskGphFromMdDiffAcc(
    const ChainNode*) const {
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    return &TaskGraph::BldSubTskGphByBoxing;
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    UNEXPECTED_RUN();
  }
}
BldBoxingOpMthd MdUpdtChainNode::GetMthdForBldBoxingOpFromMdDiffAcc(
    const ChainNode*) const {
  return &BoxingTaskNode::BldBoxingOpWithAddClone;
}
std::vector<std::string> MdUpdtChainNode::FindLbnsFromMdDiffAcc(
    const ChainNode*) const {
  return {kPackedBlobName};
}

// MdSaveChainNode
BldSubTskGphMthd MdSaveChainNode::GetMthdForBldSubTskGphFromMdUpdt(
    const ChainNode*) const {
  if (parallel_desc()->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
}

// MdDiffAccChainNode
BldSubTskGphMthd MdDiffAccChainNode::GetMthdForBldSubTskGphFromBackward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

std::vector<std::string> FindLbnsBetween(const ChainNode* in_chain,
                                         const ChainNode* out_chain) {
  return in_chain->FindLbnsTo(out_chain);
}

std::string ChainEdge::VisualStr() const { return ""; }

BldSubTskGphMthd ChainEdge::GetMthdForBldSubTskGph() const {
  return src_node()->GetMthdForBldSubTskGphTo(dst_node());
}

}  // namespace oneflow
