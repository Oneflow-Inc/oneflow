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
#define BOXING_METHOD_TYPE_TUPLE_SEQ \
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
//  BOXING_METHOD_TYPE_TUPLE_SEQ is equivalent to the above.
#define BOXING_METHOD_TYPE_TUPLE_SEQ                                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                           \
      COMBINE_PARALLEL_POLICY, (true), (FORWARD_METHOD_4_TUPLE),              \
      FAR_FROM_LOSS_POLICY_MTHD_TYPE_SEQ, CLOSE_TO_LOSS_POLICY_MTHD_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(                                           \
      COMBINE_PARALLEL_POLICY, (false), (BACKWARD_METHOD_4_TUPLE),            \
      CLOSE_TO_LOSS_POLICY_MTHD_TYPE_SEQ, FAR_FROM_LOSS_POLICY_MTHD_TYPE_SEQ)

namespace oneflow {

namespace {

#define FW_OR_BW_LBNS_FUNC_SEQ                          \
  OF_PP_MAKE_TUPLE_SEQ(Fw, true, output_bns, input_bns) \
  OF_PP_MAKE_TUPLE_SEQ(Bw, false, input_diff_bns, output_diff_bns)

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
                       BOXING_METHOD_TYPE_TUPLE_SEQ);

  LOG(FATAL) << (is_forward ? "in " : "out_diff ") << in_policy
             << (is_forward ? " out " : " in_diff ") << out_policy;
  return nullptr;
}

#define CONF_METHOD_4_CHAIN_PAIR(fw_or_bw, is_forward, out_fn, in_fn)       \
  BldBoxingOpConfMthd GetBldBoxingOpConfMethodBy##fw_or_bw##ParallelPolicy( \
      const ChainNode* in_chain, const ChainNode* out_chain) {              \
    return GetBldBoxingOpConfMthd(is_forward, in_chain, out_chain);         \
  }
OF_PP_FOR_EACH_TUPLE(CONF_METHOD_4_CHAIN_PAIR, FW_OR_BW_LBNS_FUNC_SEQ);

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

#define FIND_LBNS_BETWEEN(fw_or_bw, is_forward, out_bns, in_bns)             \
  std::vector<std::string> FindLbnsBetween##fw_or_bw(                        \
      const ChainNode* in_chain, const ChainNode* out_chain) {               \
    return FindLbnsBetweenChainPair(in_chain, &Operator::out_bns, out_chain, \
                                    &Operator::in_bns);                      \
  }
OF_PP_FOR_EACH_TUPLE(FIND_LBNS_BETWEEN, FW_OR_BW_LBNS_FUNC_SEQ);

BldSubTskGphMthd GetMthdForBldSubTskGphByParellPolicy(
    const MdUpdtChainNode* mdupdt) {
  ParallelPolicy policy = mdupdt->parallel_desc()->policy();
  if (policy == ParallelPolicy::kDataParallel) {
    return &TaskGraph::BldSubTskGphByBoxing;
  } else if (policy == ParallelPolicy::kModelParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    UNEXPECTED_RUN();
  }
}

BldSubTskGphMthd GetMthdForBldSubTskGphByParallelNum(
    const MdSaveChainNode* mdsave) {
  if (mdsave->parallel_desc()->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
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

#define CHAIN_NODE_FUNC_INFO_SEQ                                          \
  OF_PP_SEQ_PRODUCT((ForwardChainNode), FORWARD_CHAIN_NODE_FUNC_SEQ)      \
  OF_PP_SEQ_PRODUCT((BackwardChainNode), BACKWARD_CHAIN_NODE_FUNC_SEQ)    \
  OF_PP_SEQ_PRODUCT((LossChainNode), LOSS_CHAIN_NODE_FUNC_SEQ)            \
  OF_PP_SEQ_PRODUCT((LossAccChainNode), LOSS_ACC_CHAIN_NODE_FUNC_SEQ)     \
  OF_PP_SEQ_PRODUCT((LossPrintChainNode), LOSS_PRINT_CHAIN_NODE_FUNC_SEQ) \
  OF_PP_SEQ_PRODUCT((MdUpdtChainNode), MDUPDT_CHAIN_NODE_FUNC_SEQ)        \
  OF_PP_SEQ_PRODUCT((MdSaveChainNode), MDSAVE_CHAIN_NODE_FUNC_SEQ)        \
  OF_PP_SEQ_PRODUCT((MdDiffAccChainNode), MDDIFF_ACC_CHAIN_NODE_FUNC_SEQ)

#define DEF_CHAIN_NODE_FUNC(ret_type, class_name, func_name, ret_value) \
  ret_type class_name::func_name(const ChainNode* node) const {         \
    return ret_value;                                                   \
  }

#define DEFINE_CHAIN_NODE_FUNC(class_name, t)                                  \
  DEF_CHAIN_NODE_FUNC(                                                         \
      OF_PP_TUPLE_ELEM(0, t), class_name,                                      \
      OF_PP_CAT(OF_PP_TUPLE_ELEM(1, t),                                        \
                OF_PP_CAT(From, OF_PP_TUPLE_ELEM(0, OF_PP_TUPLE_ELEM(2, t)))), \
      OF_PP_TUPLE_ELEM(1, OF_PP_TUPLE_ELEM(2, t)))

OF_PP_FOR_EACH_TUPLE(DEFINE_CHAIN_NODE_FUNC, CHAIN_NODE_FUNC_INFO_SEQ)

void ForwardChainNode::set_data_output_lbns() {
  ForEachNodeOnOutEdge([this](const ChainNode* to_node) {
    if (dynamic_cast<const ForwardChainNode*>(to_node)
        || dynamic_cast<const LossChainNode*>(to_node)) {
      AddDataOutputLbnsTo(to_node);
    }
  });
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

std::vector<std::string> FindLbnsBetween(const ChainNode* in_chain,
                                         const ChainNode* out_chain) {
  return in_chain->FindLbnsTo(out_chain);
}

std::string ChainEdge::VisualStr() const { return ""; }

BldSubTskGphMthd ChainEdge::GetMthdForBldSubTskGph() const {
  return src_node()->GetMthdForBldSubTskGphTo(dst_node());
}

}  // namespace oneflow
