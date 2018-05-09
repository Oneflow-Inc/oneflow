#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/normal_backward_compute_task_node.h"
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
#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

namespace {

BldBoxingOpConfMthd GetBldBoxingOpConfMethodByFwParallelPolicy(const LogicalNode* in_logical,
                                                               const LogicalNode* out_logical) {
  ParallelPolicy in_policy = in_logical->parallel_desc()->policy();
  ParallelPolicy out_policy = out_logical->parallel_desc()->policy();
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
  return nullptr;
}
BldBoxingOpConfMthd GetBldBoxingOpConfMethodByBwParallelPolicy(const LogicalNode* in_logical,
                                                               const LogicalNode* out_logical) {
  ParallelPolicy in_policy = in_logical->parallel_desc()->policy();
  ParallelPolicy out_policy = out_logical->parallel_desc()->policy();
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
  return nullptr;
}

std::string ConcatTypeName(const LogicalNode* lhs, const LogicalNode* rhs) {
  return lhs->TypeName() + rhs->TypeName();
}

using FuncForFindBldSubTskGphMthd =
    std::function<BldSubTskGphMthd(const LogicalNode* src, const LogicalNode* dst)>;

DEFINE_STATIC_VAR(HashMap<std::string OF_COMMA FuncForFindBldSubTskGphMthd>,
                  GetFuncForFindBldSubTskGphMthd);

void AddFuncForFindBldSubTskGphMthd(const std::string& k, FuncForFindBldSubTskGphMthd v) {
  CHECK(GetFuncForFindBldSubTskGphMthd()->emplace(k, v).second);
}
void AddFuncForFindBldSubTskGphMthd(const std::string& k, std::function<BldSubTskGphMthd()> v) {
  AddFuncForFindBldSubTskGphMthd(k, [v](const LogicalNode*, const LogicalNode*) { return v(); });
}
void AddFuncForFindBldSubTskGphMthd(const std::string& k, BldSubTskGphMthd v) {
  AddFuncForFindBldSubTskGphMthd(k, [v](const LogicalNode*, const LogicalNode*) { return v; });
}
#define REGISTER_BLD_SUB_TSK_GPH_MTHD(k, v) COMMAND(AddFuncForFindBldSubTskGphMthd(k, v))

using FuncForFindBldBoxingOpConfMthd =
    std::function<BldBoxingOpConfMthd(const LogicalNode* src, const LogicalNode* dst)>;
DEFINE_STATIC_VAR(HashMap<std::string OF_COMMA FuncForFindBldBoxingOpConfMthd>,
                  GetFuncForFindBldBoxingOpConfMthd);

void AddFuncForFindBldBoxingOpConfMthd(const std::string& k, FuncForFindBldBoxingOpConfMthd v) {
  CHECK(GetFuncForFindBldBoxingOpConfMthd()->emplace(k, v).second);
}
void AddFuncForFindBldBoxingOpConfMthd(const std::string& k,
                                       std::function<BldBoxingOpConfMthd()> v) {
  AddFuncForFindBldBoxingOpConfMthd(k, [v](const LogicalNode*, const LogicalNode*) { return v(); });
}
void AddFuncForFindBldBoxingOpConfMthd(const std::string& k, BldBoxingOpConfMthd v) {
  AddFuncForFindBldBoxingOpConfMthd(k, [v](const LogicalNode*, const LogicalNode*) { return v; });
}
#define REGISTER_BLD_BOXING_OP_CONF_MTHD(k, v) COMMAND(AddFuncForFindBldBoxingOpConfMthd(k, v))

BldSubTskGphMthd BldSubTskGphToMdSave(const LogicalNode*, const LogicalNode* save) {
  if (save->parallel_desc()->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
}

BldSubTskGphMthd BldSubTskGphToNormalMdUpdt(const LogicalNode*, const LogicalNode* updt) {
  if (updt->parallel_desc()->policy() == kDataParallel) {
    return &TaskGraph::BldSubTskGphByBoxing;
  } else if (updt->parallel_desc()->policy() == kModelParallel) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  } else {
    UNIMPLEMENTED();
  }
}

using FuncForFindLbis =
    std::function<std::vector<LogicalBlobId>(const LogicalNode* src, const LogicalNode* dst)>;
DEFINE_STATIC_VAR(HashMap<std::string OF_COMMA FuncForFindLbis>, GetFuncForFindLbis);

#define REGISTER_FUNC_FOR_FIND_LBIS(k, v) COMMAND(CHECK(GetFuncForFindLbis()->emplace(k, v).second))

std::vector<LogicalBlobId> ReturnPackedLbi(const LogicalNode* src, const LogicalNode* dst) {
  return {GenPackedLbi()};
}

REGISTER_FUNC_FOR_FIND_LBIS("LossAcc"
                            "LossPrint",
                            ReturnPackedLbi);
REGISTER_FUNC_FOR_FIND_LBIS("MdDiffAcc"
                            "NormalMdUpdt",
                            ReturnPackedLbi);
REGISTER_FUNC_FOR_FIND_LBIS("NormalBackward"
                            "NormalMdUpdt",
                            ReturnPackedLbi);

}  // namespace

std::shared_ptr<Operator> LogicalNode::SoleOp() const {
  CHECK_EQ(op_vec_.size(), 1);
  return op_vec_.front();
}

std::vector<LogicalBlobId> LogicalNode::GetLbisTo(const LogicalNode* dst) const {
  auto it = dst2data_lbis_.find(dst);
  if (it != dst2data_lbis_.end()) {
    return it->second;
  } else {
    std::string k = ConcatTypeName(this, dst);
    auto func_it = GetFuncForFindLbis()->find(k);
    CHECK(func_it != GetFuncForFindLbis()->end()) << k;
    return func_it->second(this, dst);
  }
}

void LogicalNode::SetDataLbisTo(const LogicalNode* dst, const std::vector<LogicalBlobId>& lbis) {
  CHECK(dst2data_lbis_.emplace(dst, lbis).second);
  BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(this, dst);
  if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
    lbi_boxing_.insert(lbis.begin(), lbis.end());
  } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
    lbi_121_.insert(lbis.begin(), lbis.end());
  } else {
    UNIMPLEMENTED();
  }
}

std::string LogicalNode::VisualStr() const {
  std::stringstream ss;
  ss << TypeName();
  for (std::shared_ptr<Operator> op : op_vec_) { ss << "\\n" << op->op_name(); }
  return ss.str();
}

bool LogicalNode::HasOpWithModelOrModelTmpBlob() const {
  return HasOpWithCondition([](const Operator* op) {
    return op->model_bns().empty() == false || op->model_tmp_bns().empty() == false;
  });
}

bool LogicalNode::HasOpWithModelBlob() const {
  return HasOpWithCondition([](const Operator* op) { return op->model_bns().empty() == false; });
}

bool LogicalNode::HasOpWithForwardModelBlob() const {
  return HasOpWithCondition(
      [](const Operator* op) { return op->forward_model_bns().empty() == false; });
}

void LogicalNode::GenSortedCompTaskNodes(std::function<int64_t(const TaskNode*)> AllocateCpuThrdId,
                                         std::function<void(CompTaskNode*)> Handler) const {
  int64_t parallel_idx = 0;
  int64_t parallel_num = parallel_desc_->parallel_num();
  for (int64_t machine_id : parallel_desc_->sorted_machine_ids()) {
    for (int64_t dev_phy_id : parallel_desc_->sorted_dev_phy_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode();
      comp_task_node->set_machine_id(machine_id);
      if (parallel_desc_->device_type() == DeviceType::kGPU) {
        comp_task_node->set_thrd_id(Global<IDMgr>::Get()->GetGpuDeviceThrdId(dev_phy_id));
      } else if (parallel_desc_->device_type() == DeviceType::kCPU) {
        comp_task_node->set_thrd_id(AllocateCpuThrdId(comp_task_node));
      } else {
        UNIMPLEMENTED();
      }
      comp_task_node->set_logical_node(this);
      comp_task_node->mut_parallel_ctx()->set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx()->set_parallel_num(parallel_num);
      comp_task_node->mut_parallel_ctx()->set_policy(parallel_desc_->policy());
      FixCompTaskNode(comp_task_node);
      Handler(comp_task_node);
    }
  }
}

int32_t LogicalNode::GetModelSplitAxis() const {
  CHECK_EQ(parallel_desc_->policy(), kModelParallel);
  CHECK_NOTNULL(main_model_parallel_);
  if (main_model_parallel_ == this) {
    int32_t ret = SoleOp()->ModelSplitAxis();
    CHECK_NE(ret, -1);
    return ret;
  } else {
    return main_model_parallel_->GetModelSplitAxis();
  }
}

int32_t LogicalNode::GetMaxModelSplitNum() const {
  CHECK_EQ(parallel_desc_->policy(), kModelParallel);
  CHECK_NOTNULL(main_model_parallel_);
  if (main_model_parallel_ == this) {
    int32_t ret = SoleOp()->MaxModelSplitNum();
    CHECK_NE(ret, -1);
    return ret;
  } else {
    return main_model_parallel_->GetMaxModelSplitNum();
  }
}

bool LogicalNode::HasOpWithCondition(std::function<bool(const Operator*)> cond) const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (cond(op.get())) { return true; }
  }
  return false;
}

static bool IsModelParallel121(const LogicalNode* src_node, const LogicalNode* dst_node) {
  return src_node->main_model_parallel() == dst_node->main_model_parallel();
}

BldSubTskGphMthd GetMthdForBldSubTskGph(const LogicalNode* src_node, const LogicalNode* dst_node) {
  std::shared_ptr<const ParallelDesc> src_pd = src_node->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst_node->parallel_desc();
  if (src_pd->parallel_num() == 1 && dst_pd->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
  std::string k = ConcatTypeName(src_node, dst_node);
  auto it = GetFuncForFindBldSubTskGphMthd()->find(k);
  if (it != GetFuncForFindBldSubTskGphMthd()->end()) { return it->second(src_node, dst_node); }
  if (src_pd->parallel_num() == dst_pd->parallel_num()) {
    if (src_pd->policy() == kDataParallel && dst_pd->policy() == kDataParallel) {
      return &TaskGraph::BldSubTskGphByOneToOne;
    } else if (src_pd->policy() == kModelParallel && dst_pd->policy() == kModelParallel
               && IsModelParallel121(src_node, dst_node)) {
      return &TaskGraph::BldSubTskGphByOneToOne;
    } else {
      // do nothing
    }
  }
  return &TaskGraph::BldSubTskGphByBoxing;
}

REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalMdUpdt"
                              "NormalForward",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalMdUpdt"
                              "NormalBackward",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalMdUpdt"
                              "MdSave",
                              BldSubTskGphToMdSave);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalForward"
                              "MdSave",
                              BldSubTskGphToMdSave);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalForward"
                              "NormalBackward",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalBackward"
                              "MdDiffAcc",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("RecordLoad"
                              "Decode",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("Loss"
                              "LossAcc",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("MdDiffAcc"
                              "NormalMdUpdt",
                              BldSubTskGphToNormalMdUpdt);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalBackward"
                              "NormalMdUpdt",
                              BldSubTskGphToNormalMdUpdt);
REGISTER_BLD_SUB_TSK_GPH_MTHD("MdDiffAcc"
                              "ReduceScatter",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceScatter"
                              "ReduceAdd",
                              &TaskGraph::BldSubTskGphByReduceScatter2ReduceAdd);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceAdd"
                              "ReduceGather",
                              &TaskGraph::BldSubTskGphByReduceAdd2ReduceGather);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceGather"
                              "NormalMdUpdt",
                              &TaskGraph::BldSubTskGphByOneToOne);

BldBoxingOpConfMthd GetMthdForBldBoxingOpConf(const LogicalNode* src, const LogicalNode* dst) {
  std::string k = ConcatTypeName(src, dst);
  auto it = GetFuncForFindBldBoxingOpConfMthd()->find(k);
  if (it != GetFuncForFindBldBoxingOpConfMthd()->end()) { return it->second(src, dst); }
  return GetBldBoxingOpConfMethodByFwParallelPolicy(src, dst);
}

REGISTER_BLD_BOXING_OP_CONF_MTHD("NormalBackward"
                                 "NormalBackward",
                                 &GetBldBoxingOpConfMethodByBwParallelPolicy);
REGISTER_BLD_BOXING_OP_CONF_MTHD("Loss"
                                 "NormalBackward",
                                 &GetBldBoxingOpConfMethodByBwParallelPolicy);
REGISTER_BLD_BOXING_OP_CONF_MTHD("LossAcc"
                                 "LossPrint",
                                 &BoxingTaskNode::BldBoxingOpConfWithAddAndClone);
REGISTER_BLD_BOXING_OP_CONF_MTHD("MdDiffAcc"
                                 "NormalMdUpdt",
                                 &BoxingTaskNode::BldBoxingOpConfWithAddAndClone);
REGISTER_BLD_BOXING_OP_CONF_MTHD("NormalBackward"
                                 "NormalMdUpdt",
                                 &BoxingTaskNode::BldBoxingOpConfWithAddAndClone);

#define LOGICAL_TYPE_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(NormalForward)  \
  OF_PP_MAKE_TUPLE_SEQ(NormalBackward) \
  OF_PP_MAKE_TUPLE_SEQ(RecordLoad)     \
  OF_PP_MAKE_TUPLE_SEQ(Decode)         \
  OF_PP_MAKE_TUPLE_SEQ(Loss)           \
  OF_PP_MAKE_TUPLE_SEQ(LossAcc)        \
  OF_PP_MAKE_TUPLE_SEQ(LossPrint)      \
  OF_PP_MAKE_TUPLE_SEQ(NormalMdUpdt)   \
  OF_PP_MAKE_TUPLE_SEQ(MdSave)         \
  OF_PP_MAKE_TUPLE_SEQ(MdDiffAcc)      \
  OF_PP_MAKE_TUPLE_SEQ(Print)          \
  OF_PP_MAKE_TUPLE_SEQ(ReduceScatter)  \
  OF_PP_MAKE_TUPLE_SEQ(ReduceAdd)      \
  OF_PP_MAKE_TUPLE_SEQ(ReduceGather)

#define DEFINE_VIRTUAL_METHOD(x)                              \
  std::string x##LogicalNode::TypeName() const { return #x; } \
  CompTaskNode* x##LogicalNode::NewCompTaskNode() const { return new x##CompTaskNode; }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, LOGICAL_TYPE_SEQ);

BackwardLogicalNode* ForwardLogicalNode::NewBackwardNode() {
  bw_node_ = NewCorrectBackwardNode();
  bw_node_->mut_op_vec() = op_vec();
  bw_node_->mut_parallel_desc() = parallel_desc();
  bw_node_->fw_node_ = this;
  bw_node_->set_main_model_parallel(main_model_parallel());
  return bw_node_;
}

BackwardLogicalNode* NormalForwardLogicalNode::NewCorrectBackwardNode() {
  return new NormalBackwardLogicalNode;
}

void NormalMdUpdtLogicalNode::FixCompTaskNode(CompTaskNode* node) const {
  NormalMdUpdtCompTaskNode* normal_mdupdt_node = static_cast<NormalMdUpdtCompTaskNode*>(node);
  if (parallel_desc()->policy() == ParallelPolicy::kDataParallel) {
    normal_mdupdt_node->set_random_seed(random_seed_);
  } else if (parallel_desc()->policy() == ParallelPolicy::kModelParallel) {
    normal_mdupdt_node->set_random_seed(NewRandomSeed());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
