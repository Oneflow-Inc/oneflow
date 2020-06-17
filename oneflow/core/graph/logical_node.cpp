#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/optimizer_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/print_compute_task_node.h"
#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/decode_random_compute_task_node.h"
#include "oneflow/core/graph/distribute_concat_compute_task_node.h"
#include "oneflow/core/graph/distribute_split_compute_task_node.h"
#include "oneflow/core/graph/record_load_compute_task_node.h"
#include "oneflow/core/graph/reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/reduce_add_compute_task_node.h"
#include "oneflow/core/graph/reduce_gather_compute_task_node.h"
#include "oneflow/core/graph/reduce_concat_compute_task_node.h"
#include "oneflow/core/graph/reduce_split_compute_task_node.h"
#include "oneflow/core/graph/nccl_all_reduce_compute_task_node.h"
#include "oneflow/core/graph/nccl_reduce_scatter_compute_task_node.h"
#include "oneflow/core/graph/nccl_all_gather_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/reduce_identity_task_node.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

namespace {

const LogicalEdge* GetConnectedEdge(const LogicalNode* src_node, const LogicalNode* dst_node) {
  LogicalEdge* connect_edge = nullptr;
  for (LogicalEdge* edge : src_node->out_edges()) {
    if (edge->dst_node() == dst_node) {
      CHECK(connect_edge == nullptr);
      connect_edge = edge;
    }
  }
  return connect_edge;
}

static bool IsConnectedLbisAllSameSbpParallel(const LogicalNode* src_node,
                                              const LogicalNode* dst_node) {
  if (src_node->parallel_desc()->parallel_num() != dst_node->parallel_desc()->parallel_num()) {
    return false;
  }
  const LogicalEdge* connect_edge = GetConnectedEdge(src_node, dst_node);
  CHECK_NOTNULL(connect_edge);
  CHECK_GT(connect_edge->lbis().size(), 0);
  const std::string& src_op_name = src_node->SoleOp()->op_name();
  const std::string& dst_op_name = dst_node->SoleOp()->op_name();
  HashSet<bool> predicators;
  for (const LogicalBlobId& lbi : connect_edge->lbis()) {
    const auto& src_sbp = Global<OpGraph>::Get()->GetSbpParallel(src_op_name, lbi);
    const auto& dst_sbp = Global<OpGraph>::Get()->GetSbpParallel(dst_op_name, lbi);
    predicators.insert(src_sbp == dst_sbp);
  }
  CHECK_EQ(predicators.size(), 1);
  return *predicators.begin();
}

static bool IsProducedLbisAllBroadcastParallel(const LogicalNode* src_node,
                                               const LogicalNode* dst_node) {
  const LogicalEdge* connect_edge = GetConnectedEdge(src_node, dst_node);
  CHECK_NOTNULL(connect_edge);
  CHECK_GT(connect_edge->lbis().size(), 0);
  const std::string& src_op_name = src_node->SoleOp()->op_name();
  HashSet<bool> predicators;
  for (const LogicalBlobId& lbi : connect_edge->lbis()) {
    const auto& src_sbp = Global<OpGraph>::Get()->GetSbpParallel(src_op_name, lbi);
    predicators.insert(src_sbp.has_broadcast_parallel());
  }
  CHECK_EQ(predicators.size(), 1);
  return *predicators.begin();
}

bool HasSoleIdentityOp(const LogicalNode* logical_node) {
  const auto& op_conf = logical_node->SoleOp()->op_conf();
  return logical_node->op_vec().size() == 1 && op_conf.has_tuple_identity_conf();
}

BldBoxingOpConfMthd GetBldBoxingOpConfMethodByFwParallelPolicy(const LogicalNode* in_logical,
                                                               const LogicalNode* out_logical) {
  return &BoxingTaskNode::BldBoxingOpConfWithFwSbpParallel;
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

BldSubTskGphMthd BldSubTskGphToNormalMdUpdt(const LogicalNode*, const LogicalNode* updt) {
  TODO();  // outdate
}

using FuncForFindLbis =
    std::function<std::vector<LogicalBlobId>(const LogicalNode* src, const LogicalNode* dst)>;
DEFINE_STATIC_VAR(HashMap<std::string OF_COMMA FuncForFindLbis>, GetFuncForFindLbis);

#define REGISTER_FUNC_FOR_FIND_LBIS(k, v) COMMAND(CHECK(GetFuncForFindLbis()->emplace(k, v).second))

std::vector<LogicalBlobId> ReturnPackedLbi(const LogicalNode* src, const LogicalNode* dst) {
  return {GenPackedLbi()};
}

REGISTER_FUNC_FOR_FIND_LBIS("MdDiffAcc"
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
}

bool LogicalNode::IsDataLbiOnOutEdge(const LogicalBlobId& lbi) const {
  for (const auto& pair : dst2data_lbis_) {
    if (std::find(pair.second.begin(), pair.second.end(), lbi) != pair.second.end()) {
      return true;
    }
  }
  return false;
}

std::string LogicalNode::VisualStr() const {
  std::stringstream ss;
  ss << TypeName();
  for (std::shared_ptr<Operator> op : op_vec_) { ss << "\\n" << op->op_name(); }
  return ss.str();
}

void LogicalNode::GenSortedCompTaskNodes(
    std::function<int64_t(const TaskNode*)> AllocateCpuThrdIdEvenly,
    std::vector<std::pair<int64_t, CompTaskNode*>>* nodes,
    std::function<void(CompTaskNode*)> Handler) const {
  int64_t parallel_idx = 0;
  int64_t parallel_num = parallel_desc_->parallel_num();
  for (int64_t machine_id : parallel_desc_->sorted_machine_ids()) {
    for (int64_t dev_phy_id : parallel_desc_->sorted_dev_phy_ids(machine_id)) {
      CompTaskNode* comp_task_node = NewCompTaskNode();
      comp_task_node->set_machine_id(machine_id);
      comp_task_node->mut_parallel_ctx()->set_parallel_id(parallel_idx++);
      comp_task_node->mut_parallel_ctx()->set_parallel_num(parallel_num);

      const IDMgr* id_mgr = Global<IDMgr>::Get();
      if (parallel_desc_->device_type() == DeviceType::kGPU) {
        switch (comp_task_node->GetCudaWorkType()) {
          case CudaWorkType::kCompute: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuComputeThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kCopyH2D: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuH2DThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kCopyD2H: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuD2HThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kNccl: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuNcclThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kMix: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuMixThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kReduceCtrl: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuReduceCtrlThrdId(dev_phy_id));
            break;
          }
          case CudaWorkType::kMdUpdt: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuMdUpdtThrdId(dev_phy_id));
            break;
          }
          default: UNIMPLEMENTED();
        }
      } else if (parallel_desc_->device_type() == DeviceType::kCPU) {
        if (comp_task_node->IsIndependent()) {
          nodes->push_back({machine_id, comp_task_node});
        } else {
          comp_task_node->set_thrd_id(AllocateCpuThrdIdEvenly(comp_task_node));
        }
      } else {
        UNIMPLEMENTED();
      }
      comp_task_node->set_logical_node(this);
      FixCompTaskNode(comp_task_node);
      Handler(comp_task_node);
    }
  }
}

bool LogicalNode::HasOpWithCondition(std::function<bool(const Operator*)> cond) const {
  for (std::shared_ptr<const Operator> op : op_vec_) {
    if (cond(op.get())) { return true; }
  }
  return false;
}

BldSubTskGphMthd GetMthdForBldSubTskGph(const LogicalNode* src_node, const LogicalNode* dst_node) {
  std::shared_ptr<const ParallelDesc> src_pd = src_node->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst_node->parallel_desc();
  if (src_node->op_vec().size() == 1 && dst_node->op_vec().size() == 1) {
    if (src_node->SoleOp()->op_conf().has_record_load_conf()
        && dst_node->SoleOp()->op_conf().has_tick_conf()) {
      CHECK(src_pd->parallel_num() == dst_pd->parallel_num());
    }
    auto IsTickNode = [&](const LogicalNode* node) {
      return IsClassRegistered<IsTickTockOpTypeCase>(node->SoleOp()->op_conf().op_type_case());
    };
    if (IsTickNode(src_node) || IsTickNode(dst_node)) {
      if (src_pd->parallel_num() > 1 && dst_pd->parallel_num() == 1
          && src_node->SoleOp()->op_conf().has_partial_tick_conf()) {
        CHECK(dst_node->SoleOp()->op_conf().has_sink_tick_conf());
        return &TaskGraph::BldSubTskGphByBoxing;
      } else {
        if (IsTickNode(src_node) && IsTickNode(dst_node)) {
          if (src_pd->parallel_num() > 1) {
            CHECK_EQ(src_pd->parallel_num(), dst_pd->parallel_num());
          }
        }
        return &TaskGraph::BldSubTskGphByBroadcastToBroadcast;
      }
    }
  }
  if (src_pd->parallel_num() == 1 && dst_pd->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
  std::string k = ConcatTypeName(src_node, dst_node);
  auto it = GetFuncForFindBldSubTskGphMthd()->find(k);
  if (it == GetFuncForFindBldSubTskGphMthd()->end()) {
    it = GetFuncForFindBldSubTskGphMthd()->find(src_node->TypeName() + "*");
  }
  if (it == GetFuncForFindBldSubTskGphMthd()->end()) {
    it = GetFuncForFindBldSubTskGphMthd()->find("*" + dst_node->TypeName());
  }
  if (it != GetFuncForFindBldSubTskGphMthd()->end()) { return it->second(src_node, dst_node); }
  if (src_pd->parallel_num() == dst_pd->parallel_num()
      && IsConnectedLbisAllSameSbpParallel(src_node, dst_node)) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
  if (IsProducedLbisAllBroadcastParallel(src_node, dst_node) && !GlobalJobDesc().use_boxing_v2()) {
    return &TaskGraph::BldSubTskGphBySelectOneSourceToSoleSink;
  } else {
    return &TaskGraph::BldSubTskGphByBoxing;
  }
}

REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalMdUpdt"
                              "NormalForward",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalForward"
                              "ReduceConcat",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceConcat"
                              "ReduceIdentity",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalForward"
                              "ReduceIdentity",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceIdentity"
                              "NcclAllReduce",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceIdentity"
                              "ReduceScatter",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceIdentity"
                              "NcclReduceScatter",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NcclAllReduce"
                              "ReduceSplit",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceSplit"
                              "NormalForward",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("RecordLoad"
                              "Decode",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("MdDiffAcc"
                              "NormalMdUpdt",
                              BldSubTskGphToNormalMdUpdt);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceScatter"
                              "ReduceAdd",
                              &TaskGraph::BldSubTskGphByReduceScatter2ReduceAdd);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceAdd"
                              "ReduceGather",
                              &TaskGraph::BldSubTskGphByReduceAdd2ReduceGather);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceGather"
                              "ReduceGather",
                              &TaskGraph::BldSubTskGphByReduceGather2ReduceGather);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceGather"
                              "ReduceSplit",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("NcclAllGather"
                              "ReduceSplit",
                              &TaskGraph::BldSubTskGphByOneToOne);
REGISTER_BLD_SUB_TSK_GPH_MTHD("ReduceAdd"
                              "ReduceScatter",
                              &TaskGraph::BldSubTskGphByOneToOne);

REGISTER_BLD_SUB_TSK_GPH_MTHD("*"
                              "DistributeConcat",
                              &TaskGraph::BldSubTskGphByPartialInLbiConnect);

REGISTER_BLD_SUB_TSK_GPH_MTHD("DistributeSplit"
                              "*",
                              &TaskGraph::BldSubTskGphByPartialOutLbiConnect);

BldBoxingOpConfMthd GetMthdForBldBoxingOpConf(const LogicalNode* src, const LogicalNode* dst) {
  std::string k = ConcatTypeName(src, dst);
  auto it = GetFuncForFindBldBoxingOpConfMthd()->find(k);
  if (it != GetFuncForFindBldBoxingOpConfMthd()->end()) { return it->second(src, dst); }
  return GetBldBoxingOpConfMethodByFwParallelPolicy(src, dst);
}

REGISTER_BLD_BOXING_OP_CONF_MTHD("MdDiffAcc"
                                 "NormalMdUpdt",
                                 &BoxingTaskNode::BldBoxingOpConfWithAddAndClone);
REGISTER_BLD_BOXING_OP_CONF_MTHD("Tick"
                                 "Tick",
                                 &BoxingTaskNode::BldBoxingOpConfWithPartialTick2SinkTick);

#define LOGICAL_TYPE_SEQ                                   \
  OF_PP_MAKE_TUPLE_SEQ(NormalForward, kDataForwardArea)    \
  OF_PP_MAKE_TUPLE_SEQ(DistributeConcat, kDataForwardArea) \
  OF_PP_MAKE_TUPLE_SEQ(DistributeSplit, kDataForwardArea)  \
  OF_PP_MAKE_TUPLE_SEQ(RecordLoad, kDataPreprocessArea)    \
  OF_PP_MAKE_TUPLE_SEQ(Decode, kDataPreprocessArea)        \
  OF_PP_MAKE_TUPLE_SEQ(DecodeRandom, kDataPreprocessArea)  \
  OF_PP_MAKE_TUPLE_SEQ(MdDiffAcc, kDataForwardArea)        \
  OF_PP_MAKE_TUPLE_SEQ(Print, kPrintArea)                  \
  OF_PP_MAKE_TUPLE_SEQ(ReduceConcat, kMdUpdtArea)          \
  OF_PP_MAKE_TUPLE_SEQ(ReduceIdentity, kMdUpdtArea)        \
  OF_PP_MAKE_TUPLE_SEQ(ReduceScatter, kMdUpdtArea)         \
  OF_PP_MAKE_TUPLE_SEQ(ReduceAdd, kMdUpdtArea)             \
  OF_PP_MAKE_TUPLE_SEQ(ReduceGather, kMdUpdtArea)          \
  OF_PP_MAKE_TUPLE_SEQ(ReduceSplit, kMdUpdtArea)           \
  OF_PP_MAKE_TUPLE_SEQ(NcclAllReduce, kMdUpdtArea)         \
  OF_PP_MAKE_TUPLE_SEQ(NcclReduceScatter, kMdUpdtArea)     \
  OF_PP_MAKE_TUPLE_SEQ(NcclAllGather, kMdUpdtArea)

#define DEFINE_VIRTUAL_METHOD(x, area_type)                                             \
  std::string x##LogicalNode::TypeName() const { return #x; }                           \
  CompTaskNode* x##LogicalNode::NewCompTaskNode() const { return new x##CompTaskNode; } \
  int64_t x##LogicalNode::GetAreaId() const { return area_type; }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, LOGICAL_TYPE_SEQ);

std::string OptimizerLogicalNode::TypeName() const { return "Optimizer"; }

CompTaskNode* OptimizerLogicalNode::NewCompTaskNode() const { return new OptimizerCompTaskNode; }

int64_t OptimizerLogicalNode::GetAreaId() const { return kMdUpdtArea; }

int64_t NewAreaId() {
  static int64_t next_area_id = AreaType_ARRAYSIZE;
  return ++next_area_id;
}

int32_t ReduceIdentityLogicalNode::order_in_logical_graph() const {
  const auto& op_conf = SoleOp()->op_conf();
  CHECK(op_conf.has_reduce_identity_conf());
  if (op_conf.reduce_identity_conf().has_order_in_graph()) {
    return op_conf.reduce_identity_conf().order_in_graph();
  } else {
    return order_in_logical_graph_;
  }
}

int32_t ReduceSplitLogicalNode::order_in_logical_graph() const {
  const auto& op_conf = SoleOp()->op_conf();
  CHECK(op_conf.has_reduce_split_conf());
  if (op_conf.reduce_split_conf().has_order_in_graph()) {
    return op_conf.reduce_split_conf().order_in_graph();
  } else {
    return order_in_logical_graph_;
  }
}

}  // namespace oneflow
