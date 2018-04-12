#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"
#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/loss_print_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/graph/normal_model_update_compute_task_node.h"
#include "oneflow/core/graph/print_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

namespace {

BldBoxingOpConfMthd GetBldBoxingOpConfMethodByFwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  ParallelPolicy in_policy = in_chain->parallel_desc()->policy();
  ParallelPolicy out_policy = out_chain->parallel_desc()->policy();
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithDataConcatAndDataSplit;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithDataConcatAndClone;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithModelConcatAndDataSplit;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithModelConcatAndClone;
  } else {
    LOG(FATAL) << "in " << in_policy << " out " << out_policy;
  }
}
BldBoxingOpConfMthd GetBldBoxingOpConfMethodByBwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  ParallelPolicy in_policy = in_chain->parallel_desc()->policy();
  ParallelPolicy out_policy = out_chain->parallel_desc()->policy();
  if (in_policy == kDataParallel && out_policy == kDataParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithDataConcatAndDataSplit;
  } else if (in_policy == kDataParallel && out_policy == kModelParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithDataConcatAndModelSplit;
  } else if (in_policy == kModelParallel && out_policy == kDataParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithAddAndDataSplit;
  } else if (in_policy == kModelParallel && out_policy == kModelParallel) {
    return &BoxingTaskNode::BldBoxingOpConfWithAddAndModelSplit;
  } else {
    LOG(FATAL) << "out_diff " << in_policy << " in_diff " << out_policy;
  }
}

std::vector<std::string> FindLbnsBetweenChainPair(
    const ChainNode* in_chain,
    const std::vector<std::string>& (Operator::*GetOutLbns)() const,
    const ChainNode* out_chain,
    const std::vector<std::string>& (Operator::*GetInLbns)() const) {
  HashSet<std::string> out_lbns_in_chain;
  for (std::shared_ptr<const Operator> op : in_chain->op_vec()) {
    for (const std::string& bn_in_op : (op.get()->*GetOutLbns)()) {
      const std::string& lbn = op->Lbn4BnInOp(bn_in_op);
      CHECK(out_lbns_in_chain.insert(lbn).second);
    }
  }
  std::vector<std::string> result;
  for (std::shared_ptr<const Operator> op : out_chain->op_vec()) {
    for (const std::string& bn_in_op : (op.get()->*GetInLbns)()) {
      const std::string& lbn = op->Lbn4BnInOp(bn_in_op);
      if (out_lbns_in_chain.find(lbn) != out_lbns_in_chain.end()) {
        result.push_back(lbn);
      }
    }
  }
  SortAndRemoveDuplication(&result);
  return result;
}

std::vector<std::string> FindLbnsBetweenFw(const ChainNode* in_chain,
                                           const ChainNode* out_chain) {
  return FindLbnsBetweenChainPair(in_chain, &Operator::output_bns, out_chain,
                                  &Operator::input_bns);
}
std::vector<std::string> FindLbnsBetweenBw(const ChainNode* in_chain,
                                           const ChainNode* out_chain) {
  return FindLbnsBetweenChainPair(in_chain, &Operator::input_diff_bns,
                                  out_chain, &Operator::output_diff_bns);
}

}  // namespace

std::shared_ptr<const Operator> ChainNode::SoleOp() const {
  CHECK_EQ(op_vec_.size(), 1);
  return op_vec_.front();
}

const std::vector<std::shared_ptr<Operator>>& ChainNode::op_vec() const {
  return op_vec_;
}

bool ChainNode::HasSoleRecurrentOp() const {
  return op_vec_.size() == 1 && op_vec_.front()->IsRecurrentOp();
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
  return HasOpWithCondition([](const Operator* op) {
    return op->model_bns().empty() == false
           || op->model_tmp_bns().empty() == false;
  });
}

bool ChainNode::HasOpWithModelBlob() const {
  return HasOpWithCondition(
      [](const Operator* op) { return op->model_bns().empty() == false; });
}

bool ChainNode::HasOpWithForwardModelBlob() const {
  return HasOpWithCondition([](const Operator* op) {
    return op->forward_model_bns().empty() == false;
  });
}

void ChainNode::GenSortedCompTaskNodes(
    std::function<int64_t(const TaskNode*)> AllocateCpuThrdId,
    CompTaskNodeHandler Handler) const {
  int64_t parallel_idx = 0;
  int64_t parallel_num = parallel_desc_->parallel_num();
  for (int64_t machine_id : parallel_desc_->sorted_machine_ids()) {
    for (int64_t dev_phy_id : parallel_desc_->sorted_dev_phy_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode();
      comp_task_node->set_machine_id(machine_id);
      if (parallel_desc_->device_type() == DeviceType::kGPU) {
        comp_task_node->set_thrd_id(
            Global<IDMgr>::Get()->GetGpuDeviceThrdId(dev_phy_id));
      } else if (parallel_desc_->device_type() == DeviceType::kCPU) {
        comp_task_node->set_thrd_id(AllocateCpuThrdId(comp_task_node));
      } else {
        UNIMPLEMENTED();
      }
      comp_task_node->set_chain_node(this);
      comp_task_node->mut_parallel_ctx()->set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx()->set_parallel_num(parallel_num);
      comp_task_node->mut_parallel_ctx()->set_policy(parallel_desc_->policy());
      FixCompTaskNode(comp_task_node);
      Handler(comp_task_node);
    }
  }
}

int32_t ChainNode::GetModelSplitAxis() const {
  if (parallel_desc_->policy() == ParallelPolicy::kDataParallel) { return -1; }
  for (auto& op : op_vec_) {
    if (op->ModelSplitAxis() != -1) { return op->ModelSplitAxis(); }
  }
  return -1;
}

int32_t ChainNode::GetMaxModelSplitNum() const {
  if (parallel_desc_->policy() == ParallelPolicy::kDataParallel) { return -1; }
  for (auto& op : op_vec_) {
    if (op->MaxModelSplitNum() != -1) { return op->MaxModelSplitNum(); }
  }
  return -1;
}

#define DEFINE_VIRTUAL_METHOD(x)                                              \
  const char* x##ChainNode::TypeName() const { return #x "ChainNode"; }       \
  BldSubTskGphMthd x##ChainNode::GetMthdForBldSubTskGphTo(                    \
      const ChainNode* node) const {                                          \
    return node->GetMthdForBldSubTskGphFrom##x(this);                         \
  }                                                                           \
  BldBoxingOpConfMthd x##ChainNode::GetMthdForBldBoxingOpConfTo(              \
      const ChainNode* node) const {                                          \
    return node->GetMthdForBldBoxingOpConfFrom##x(this);                      \
  }                                                                           \
  std::vector<std::string> x##ChainNode::FindLbnsTo(const ChainNode* node)    \
      const {                                                                 \
    return node->FindLbnsFrom##x(this);                                       \
  }                                                                           \
  BldSubTskGphMthd ChainNode::GetMthdForBldSubTskGphFrom##x(const ChainNode*) \
      const {                                                                 \
    UNIMPLEMENTED();                                                          \
    return nullptr;                                                           \
  }                                                                           \
  BldBoxingOpConfMthd ChainNode::GetMthdForBldBoxingOpConfFrom##x(            \
      const ChainNode*) const {                                               \
    UNIMPLEMENTED();                                                          \
    return nullptr;                                                           \
  }                                                                           \
  std::vector<std::string> ChainNode::FindLbnsFrom##x(const ChainNode*)       \
      const {                                                                 \
    UNIMPLEMENTED();                                                          \
    return {};                                                                \
  }                                                                           \
  CompTaskNode* x##ChainNode::NewCompTaskNodeWithSameName() const {           \
    return new x##CompTaskNode;                                               \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, CHAIN_TYPE_SEQ)

void ChainNode::AddDataOutputLbnsTo(const ChainNode* to_node) {
  std::vector<std::string> lbns = FindLbnsTo(to_node);
  data_output_lbns_.insert(lbns.begin(), lbns.end());
}

bool ChainNode::HasOpWithCondition(
    std::function<bool(const Operator*)> cond) const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (cond(op.get())) { return true; }
  }
  return false;
}

// ForwardChainNode
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode* node) const {
  if (this == node && parallel_desc()->policy() == kDataParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    return &TaskGraph::BldSubTskGphByBoxing;
  }
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromDecode(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromNormalMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpConfMthd ForwardChainNode::GetMthdForBldBoxingOpConfFromForward(
    const ChainNode* node) const {
  if (this == node) { CHECK_EQ(parallel_desc()->policy(), kModelParallel); }
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd ForwardChainNode::GetMthdForBldBoxingOpConfFromDecode(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromDecode(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
void ForwardChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const ForwardChainNode*>(to_node)
        || dynamic_cast<const LossChainNode*>(to_node)
        || dynamic_cast<const PrintChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}
CompTaskNode* ForwardChainNode::NewCompTaskNode() const {
  if (HasSoleRecurrentOp()) {
    return new RecurrentForwardCompTaskNode;
  } else {
    return new NormalForwardCompTaskNode;
  }
}

// BackwardChainNode
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromBackward(
    const ChainNode* node) const {
  if (this == node && parallel_desc()->policy() == kDataParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    return &TaskGraph::BldSubTskGphByBoxing;
  }
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromNormalMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpConfMthd BackwardChainNode::GetMthdForBldBoxingOpConfFromBackward(
    const ChainNode* node) const {
  if (this == node) { CHECK_EQ(parallel_desc()->policy(), kModelParallel); }
  return GetBldBoxingOpConfMethodByBwParallelPolicy(node, this);
}
BldBoxingOpConfMthd BackwardChainNode::GetMthdForBldBoxingOpConfFromLoss(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByBwParallelPolicy(node, this);
}
std::vector<std::string> BackwardChainNode::FindLbnsFromBackward(
    const ChainNode* node) const {
  return FindLbnsBetweenBw(node, this);
}
std::vector<std::string> BackwardChainNode::FindLbnsFromLoss(
    const ChainNode* node) const {
  return FindLbnsBetweenBw(node, this);
}
void BackwardChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const BackwardChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}
CompTaskNode* BackwardChainNode::NewCompTaskNode() const {
  if (HasSoleRecurrentOp()) {
    return new RecurrentBackwardCompTaskNode;
  } else {
    return new NormalBackwardCompTaskNode;
  }
}

// DecodeChainNode
BldSubTskGphMthd DecodeChainNode::GetMthdForBldSubTskGphFromRecordLoad(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
void DecodeChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const ForwardChainNode*>(to_node)
        || dynamic_cast<const LossChainNode*>(to_node)
        || dynamic_cast<const PrintChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}

// LossChainNode
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromDecode(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpConfMthd LossChainNode::GetMthdForBldBoxingOpConfFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd LossChainNode::GetMthdForBldBoxingOpConfFromDecode(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> LossChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> LossChainNode::FindLbnsFromDecode(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}

void LossChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const BackwardChainNode*>(to_node)
        || dynamic_cast<const PrintChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}

// PrintChainNode
BldSubTskGphMthd PrintChainNode::GetMthdForBldSubTskGphFromDecode(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd PrintChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd PrintChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpConfMthd PrintChainNode::GetMthdForBldBoxingOpConfFromDecode(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd PrintChainNode::GetMthdForBldBoxingOpConfFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd PrintChainNode::GetMthdForBldBoxingOpConfFromLoss(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> PrintChainNode::FindLbnsFromDecode(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> PrintChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> PrintChainNode::FindLbnsFromLoss(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}

// LossAccChainNode
BldSubTskGphMthd LossAccChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}

// LossPrintChainNode
BldSubTskGphMthd LossPrintChainNode::GetMthdForBldSubTskGphFromLossAcc(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpConfMthd LossPrintChainNode::GetMthdForBldBoxingOpConfFromLossAcc(
    const ChainNode*) const {
  return &BoxingTaskNode::BldBoxingOpConfWithAddAndClone;
}
std::vector<std::string> LossPrintChainNode::FindLbnsFromLossAcc(
    const ChainNode*) const {
  return {kPackedBlobName};
}

// NormalMdUpdtChainNode
BldSubTskGphMthd NormalMdUpdtChainNode::GetMthdForBldSubTskGphFromMdDiffAcc(
    const ChainNode*) const {
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    return &TaskGraph::BldSubTskGphByBoxing;
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    UNIMPLEMENTED();
  }
}
BldBoxingOpConfMthd
NormalMdUpdtChainNode::GetMthdForBldBoxingOpConfFromMdDiffAcc(
    const ChainNode*) const {
  return &BoxingTaskNode::BldBoxingOpConfWithAddAndClone;
}
std::vector<std::string> NormalMdUpdtChainNode::FindLbnsFromMdDiffAcc(
    const ChainNode*) const {
  return {kPackedBlobName};
}

void NormalMdUpdtChainNode::FixCompTaskNode(CompTaskNode* node) const {
  NormalMdUpdtCompTaskNode* normal_mdupdt_node =
      static_cast<NormalMdUpdtCompTaskNode*>(node);
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    normal_mdupdt_node->set_random_seed(random_seed_);
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    normal_mdupdt_node->set_random_seed(NewRandomSeed());
  } else {
    UNIMPLEMENTED();
  }
}

// MdSaveChainNode
BldSubTskGphMthd MdSaveChainNode::GetMthdForBldSubTskGphFromNormalMdUpdt(
    const ChainNode*) const {
  if (parallel_desc()->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
}
BldSubTskGphMthd MdSaveChainNode::GetMthdForBldSubTskGphFromForward(
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
