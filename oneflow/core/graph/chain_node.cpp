#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"
#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/loss_print_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/model_save_compute_task_node.h"
#include "oneflow/core/graph/model_update_compute_task_node.h"
#include "oneflow/core/graph/source_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"

#define FAR_FROM_LOSS_POLICY_MTHD_TYPE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ(kDataParallel, DataConcat, DataSplit) \
  OF_PP_MAKE_TUPLE_SEQ(kModelParallel, ModelConcat, ModelSplit)

#define CLOSE_TO_LOSS_POLICY_MTHD_TYPE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ(kDataParallel, DataSplit, DataConcat) \
  OF_PP_MAKE_TUPLE_SEQ(kModelParallel, Clone, Add)

#define PARALLEL_POLICY_4_TUPLE(t) OF_PP_TUPLE_ELEM(0, t)
#define FORWARD_METHOD_4_TUPLE(t) OF_PP_TUPLE_ELEM(1, t)
#define BACKWARD_METHOD_4_TUPLE(t) OF_PP_TUPLE_ELEM(2, t)

#define COMBINE_PARALLEL_POLICY(is_forward, get_method, in, out)             \
  OF_PP_MAKE_TUPLE_SEQ(                                                      \
      is_forward, PARALLEL_POLICY_4_TUPLE(in), PARALLEL_POLICY_4_TUPLE(out), \
      OF_PP_CAT(BoxingTaskNode::BldBoxingOpConfWith,                         \
                OF_PP_CAT(get_method(in), OF_PP_CAT(And, get_method(out)))))
/*
#define BOXING_METHOD_TYPE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(true, kDataParallel, kDataParallel,
BoxingTaskNode::BldBoxingOpConfWithDataConcatAndDataSplit) \
  OF_PP_MAKE_TUPLE_SEQ(true, kDataParallel, kModelParallel,
BoxingTaskNode::BldBoxingOpConfWithDataConcatAndClone) \
  OF_PP_MAKE_TUPLE_SEQ(true, kModelParallel, kDataParallel,
BoxingTaskNode::BldBoxingOpConfWithModelConcatAndDataSplit) \
  OF_PP_MAKE_TUPLE_SEQ(true, kModelParallel, kModelParallel,
BoxingTaskNode::BldBoxingOpConfWithModelConcatAndClone)
  OF_PP_MAKE_TUPLE_SEQ(false, kDataParallel, kDataParallel,
BoxingTaskNode::BldBoxingOpConfWithDataConcatAndDataSplit) \
  OF_PP_MAKE_TUPLE_SEQ(false, kDataParallel, kModelParallel,
BoxingTaskNode::BldBoxingOpConfWithAddAndDataSplit) \
  OF_PP_MAKE_TUPLE_SEQ(false, kModelParallel, kDataParallel,
BoxingTaskNode::BldBoxingOpConfWithDataConcatAndModelSplit) \
  OF_PP_MAKE_TUPLE_SEQ(false, kModelParallel, kModelParallel,
BoxingTaskNode::BldBoxingOpConfWithAddAndModelSplit)
*/
//  BOXING_METHOD_TYPE_SEQ is equivalent to the above.
#define BOXING_METHOD_TYPE_SEQ                                                \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                           \
      COMBINE_PARALLEL_POLICY, (true), (FORWARD_METHOD_4_TUPLE),              \
      FAR_FROM_LOSS_POLICY_MTHD_TYPE_SEQ, CLOSE_TO_LOSS_POLICY_MTHD_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                           \
      COMBINE_PARALLEL_POLICY, (false), (BACKWARD_METHOD_4_TUPLE),            \
      CLOSE_TO_LOSS_POLICY_MTHD_TYPE_SEQ, FAR_FROM_LOSS_POLICY_MTHD_TYPE_SEQ)

namespace oneflow {

namespace {

BldBoxingOpConfMthd GetBldBoxingOpConfMthd(bool is_forward,
                                           const ChainNode* in_chain,
                                           const ChainNode* out_chain) {
  ParallelPolicy in_policy = in_chain->parallel_desc()->policy();
  ParallelPolicy out_policy = out_chain->parallel_desc()->policy();
#define PARALLEL_POLICY_METHOD_CASE_ENTRY(is_fw, in, out, method)    \
  if (is_forward == is_fw && in_policy == in && out_policy == out) { \
    return &method;                                                  \
  }
  OF_PP_FOR_EACH_TUPLE(PARALLEL_POLICY_METHOD_CASE_ENTRY,
                       BOXING_METHOD_TYPE_SEQ);

  LOG(FATAL) << (is_forward ? "in " : "out_diff ") << in_policy
             << (is_forward ? " out " : " in_diff ") << out_policy;
  return nullptr;
}

BldBoxingOpConfMthd GetBldBoxingOpConfMethodByFwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  return GetBldBoxingOpConfMthd(true, in_chain, out_chain);
}

BldBoxingOpConfMthd GetBldBoxingOpConfMethodByBwParallelPolicy(
    const ChainNode* in_chain, const ChainNode* out_chain) {
  return GetBldBoxingOpConfMthd(false, in_chain, out_chain);
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
      comp_task_node->set_thrd_id(dev_phy_id);
      comp_task_node->set_chain_node(this);
      comp_task_node->mut_parallel_ctx()->set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx()->set_parallel_num(parallel_num);
      comp_task_node->mut_parallel_ctx()->set_policy(parallel_desc_->policy());
      FixCompTaskNode(comp_task_node);
      Handler(comp_task_node);
    }
  }
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
    UNEXPECTED_RUN();                                                         \
    return nullptr;                                                           \
  }                                                                           \
  BldBoxingOpConfMthd ChainNode::GetMthdForBldBoxingOpConfFrom##x(            \
      const ChainNode*) const {                                               \
    UNEXPECTED_RUN();                                                         \
    return nullptr;                                                           \
  }                                                                           \
  std::vector<std::string> ChainNode::FindLbnsFrom##x(const ChainNode*)       \
      const {                                                                 \
    UNEXPECTED_RUN();                                                         \
    return {};                                                                \
  }                                                                           \
  CompTaskNode* x##ChainNode::NewCompTaskNode() const {                       \
    return new x##CompTaskNode;                                               \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, CHAIN_TYPE_SEQ)

void ChainNode::AddDataOutputLbnsTo(const ChainNode* to_node) {
  std::vector<std::string> lbns = FindLbnsTo(to_node);
  data_output_lbns_.insert(lbns.begin(), lbns.end());
}

// ForwardChainNode
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromSource(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd ForwardChainNode::GetMthdForBldSubTskGphFromMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpConfMthd ForwardChainNode::GetMthdForBldBoxingOpConfFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd ForwardChainNode::GetMthdForBldBoxingOpConfFromSource(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromForward(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
std::vector<std::string> ForwardChainNode::FindLbnsFromSource(
    const ChainNode* node) const {
  return FindLbnsBetweenFw(node, this);
}
void ForwardChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const ForwardChainNode*>(to_node)
        || dynamic_cast<const LossChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}

// BackwardChainNode
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromBackward(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromLoss(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd BackwardChainNode::GetMthdForBldSubTskGphFromMdUpdt(
    const ChainNode*) const {
  return &TaskGraph::BldSubTskGphByOneToOne;
}
BldBoxingOpConfMthd BackwardChainNode::GetMthdForBldBoxingOpConfFromBackward(
    const ChainNode* node) const {
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

// SourceChainNode
void SourceChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const ForwardChainNode*>(to_node)
        || dynamic_cast<const LossChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
}

// LossChainNode
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromForward(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldSubTskGphMthd LossChainNode::GetMthdForBldSubTskGphFromSource(
    const ChainNode* node) const {
  return &TaskGraph::BldSubTskGphByBoxing;
}
BldBoxingOpConfMthd LossChainNode::GetMthdForBldBoxingOpConfFromForward(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
}
BldBoxingOpConfMthd LossChainNode::GetMthdForBldBoxingOpConfFromSource(
    const ChainNode* node) const {
  return GetBldBoxingOpConfMethodByFwParallelPolicy(node, this);
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
BldBoxingOpConfMthd MdUpdtChainNode::GetMthdForBldBoxingOpConfFromMdDiffAcc(
    const ChainNode*) const {
  return &BoxingTaskNode::BldBoxingOpConfWithAddAndClone;
}
std::vector<std::string> MdUpdtChainNode::FindLbnsFromMdDiffAcc(
    const ChainNode*) const {
  return {kPackedBlobName};
}

void MdUpdtChainNode::FixCompTaskNode(CompTaskNode* node) const {
  MdUpdtCompTaskNode* mdupdt_node = static_cast<MdUpdtCompTaskNode*>(node);
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    mdupdt_node->set_random_seed(random_seed_);
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    mdupdt_node->set_random_seed(NewRandomSeed());
  } else {
    UNEXPECTED_RUN();
  }
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
