/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/print_compute_task_node.h"
#include "oneflow/core/graph/decode_random_compute_task_node.h"
#include "oneflow/core/graph/distribute_concat_compute_task_node.h"
#include "oneflow/core/graph/distribute_split_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/framework/framework.h"

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

}  // namespace

std::shared_ptr<const Operator> LogicalNode::SoleOp() const {
  CHECK_EQ(op_vec_.size(), 1);
  return op_vec_.front();
}

std::vector<LogicalBlobId> LogicalNode::GetLbisTo(const LogicalNode* dst) const {
  auto it = dst2data_lbis_.find(dst);
  CHECK(it != dst2data_lbis_.end());
  return it->second;
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
  for (std::shared_ptr<const Operator> op : op_vec_) { ss << "\\n" << op->op_name(); }
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
#ifdef WITH_CUDA
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
          case CudaWorkType::kDecodeH2D: {
            comp_task_node->set_thrd_id(id_mgr->GetGpuDecodeH2DThrdId(dev_phy_id));
            break;
          }
          default: UNIMPLEMENTED();
        }
#else
        UNIMPLEMENTED();
#endif
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
  const auto& IsSubsetTick = [](const OperatorConf tick) {
    return tick.has_src_subset_tick_conf() || tick.has_dst_subset_tick_conf();
  };
  std::shared_ptr<const ParallelDesc> src_pd = src_node->parallel_desc();
  std::shared_ptr<const ParallelDesc> dst_pd = dst_node->parallel_desc();
  if (src_node->op_vec().size() == 1 && dst_node->op_vec().size() == 1) {
    if (src_node->SoleOp()->op_conf().has_wait_and_send_ids_conf()
        && dst_node->SoleOp()->op_conf().has_reentrant_lock_conf()) {
      CHECK_EQ(src_pd->parallel_num(), 1);
      CHECK_EQ(dst_pd->parallel_num(), 1);
      return &TaskGraph::BldSubTskGphByBoxing;
    }
    auto IsTickNode = [&](const LogicalNode* node) {
      return IsClassRegistered<int32_t, IsTickTockOpTypeCase>(
          node->SoleOp()->op_conf().op_type_case());
    };
    if (IsTickNode(src_node) || IsTickNode(dst_node)) {
      const auto& src_op_conf = src_node->SoleOp()->op_conf();
      const auto& dst_op_conf = dst_node->SoleOp()->op_conf();
      if (src_op_conf.has_source_tick_conf()) {
        CHECK(dst_op_conf.has_tick_conf());
        CHECK_EQ(src_pd->parallel_num(), 1);
        CHECK_EQ(dst_pd->parallel_num(), 1);
        return &TaskGraph::BldSubTskGphByBoxing;
      } else if (dst_op_conf.has_sink_tick_conf()) {
        CHECK(src_op_conf.has_tick_conf() || src_op_conf.has_sink_tick_conf());
        CHECK_EQ(src_pd->parallel_num(), 1);
        CHECK_EQ(dst_pd->parallel_num(), 1);
        return &TaskGraph::BldSubTskGphByBoxing;
      } else if (IsSubsetTick(src_op_conf)) {
        return &TaskGraph::BldSubTskGphBySrcSubsetConnect;
      } else if (IsSubsetTick(dst_op_conf)) {
        return &TaskGraph::BldSubTskGphByDstSubsetConnect;
      } else {
        if (IsTickNode(src_node) && IsTickNode(dst_node)) {
          if (src_pd->parallel_num() == dst_pd->parallel_num()) {
            return &TaskGraph::BldSubTskGphByOneToOne;
          } else {
            CHECK_EQ(src_pd->parallel_num(), 1);
            return &TaskGraph::BldSubTskGphByBroadcastToBroadcast;
          }
        }
      }
    }
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
  if (src_pd->parallel_num() == 1 && dst_pd->parallel_num() == 1) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
  if (src_pd->parallel_num() == dst_pd->parallel_num()
      && IsConnectedLbisAllSameSbpParallel(src_node, dst_node)) {
    return &TaskGraph::BldSubTskGphByOneToOne;
  }
  return &TaskGraph::BldSubTskGphByBoxing;
}

REGISTER_BLD_SUB_TSK_GPH_MTHD("*"
                              "DistributeConcat",
                              &TaskGraph::BldSubTskGphByPartialInLbiConnect);

REGISTER_BLD_SUB_TSK_GPH_MTHD("DistributeSplit"
                              "*",
                              &TaskGraph::BldSubTskGphByPartialOutLbiConnect);

REGISTER_BLD_SUB_TSK_GPH_MTHD("NormalForward"
                              "DecodeH2D",
                              &TaskGraph::BldSubTskGphNormalForwardToDecodeH2D);

#define LOGICAL_TYPE_SEQ                                   \
  OF_PP_MAKE_TUPLE_SEQ(DistributeConcat, kDataForwardArea) \
  OF_PP_MAKE_TUPLE_SEQ(DistributeSplit, kDataForwardArea)  \
  OF_PP_MAKE_TUPLE_SEQ(DecodeRandom, kDataPreprocessArea)  \
  OF_PP_MAKE_TUPLE_SEQ(Print, kPrintArea)

#define DEFINE_VIRTUAL_METHOD(x, area_type)                                             \
  std::string x##LogicalNode::TypeName() const { return #x; }                           \
  CompTaskNode* x##LogicalNode::NewCompTaskNode() const { return new x##CompTaskNode; } \
  int64_t x##LogicalNode::GetAreaId() const { return area_type; }
OF_PP_FOR_EACH_TUPLE(DEFINE_VIRTUAL_METHOD, LOGICAL_TYPE_SEQ);

std::string NormalForwardLogicalNode::TypeName() const { return "NormalForward"; }

CompTaskNode* NormalForwardLogicalNode::NewCompTaskNode() const {
  if (this->SoleOp()->op_conf().has_user_conf()) {
    const OperatorConf& op_conf = this->SoleOp()->op_conf();
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (IsClassRegistered<std::string, UserOpCompTaskNodeCreator>(op_type_name)) {
      return std::unique_ptr<UserOpCompTaskNodeCreator>(
                 NewObj<std::string, UserOpCompTaskNodeCreator>(op_type_name))
          ->NewCompTaskNode(op_conf);
    } else {
      return new NormalForwardCompTaskNode;
    }
  } else {
    return new NormalForwardCompTaskNode;
  }
}

int64_t NormalForwardLogicalNode::GetAreaId() const {
  if (this->SoleOp()->op_conf().has_user_conf()) {
    const std::string& op_type_name = this->SoleOp()->op_conf().user_conf().op_type_name();
    if (IsClassRegistered<std::string, UserOpAreaIdCreator>(op_type_name)) {
      return std::unique_ptr<UserOpAreaIdCreator>(
                 NewObj<std::string, UserOpAreaIdCreator>(op_type_name))
          ->GetAreaId();
    } else {
      return AreaType::kDataForwardArea;
    }
  } else {
    return AreaType::kDataForwardArea;
  }
}

int64_t NewAreaId() {
  static int64_t next_area_id = AreaType_ARRAYSIZE;
  return ++next_area_id;
}

REGISTER_USER_OP_AREA_ID("sgd_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("indexed_slices_sgd_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("momentum_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("indexed_slices_momentum_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("adam_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("indexed_slices_adam_update", AreaType::kMdUpdtArea)
REGISTER_USER_OP_AREA_ID("lamb_update", AreaType::kMdUpdtArea)

}  // namespace oneflow
