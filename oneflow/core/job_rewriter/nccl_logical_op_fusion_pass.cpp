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
#include "oneflow/core/job/nd_sbp_util.h"
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

namespace {

class NcclLogicalOpFusionPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalOpFusionPass);
  NcclLogicalOpFusionPass() = default;
  ~NcclLogicalOpFusionPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

const std::string kNcclLogicalFusionOpNamePrefix = "Sys-NCCL-Logical-Fusion";

bool IsNcclLogicalOpNode(const OpNode* node) {
  if (node->op().op_conf().has_user_conf()) {
    const std::string& user_type_name = node->op().op_conf().user_conf().op_type_name();
    if (user_type_name == "_nccl_logical_all_reduce"
        || user_type_name == "_nccl_logical_reduce_scatter"
        || user_type_name == "_nccl_logical_reduce_scatter_noncontinuous"
        || user_type_name == "_nccl_logical_all_gather"
        || user_type_name == "_nccl_logical_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_s2s" || user_type_name == "_nccl_logical_send_recv"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_reduce"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_2D_same_dim0_all2all"
        || user_type_name == "_nccl_logical_2D_same_dim1_all_reduce") {
      return true;
    }
  }
  return false;
}

Maybe<void> ReplaceNcclOpsWithFusionOp(JobBuilder* job_builder,
                                       const std::vector<const OpNode*>& nccl_ops) const {
  if (nccl_ops.size() <= 1) { return Maybe<void>::Ok(); }
  const int32_t nccl_size = nccl_ops.size();
  const OpNode* first_nccl = nccl_ops.front();
  const ParallelDesc& seed_placement = first_nccl->parallel_desc();
  const int64_t scope_symbol_id = first_nccl->op().op_conf().scope_symbol_id();
  std::vector<std::string> src_nd_sbp_str_list;
  std::vector<std::string> dst_nd_sbp_str_list;
  auto& fusion_builder = user_op::UserOpConfWrapperBuilder("Sys-NCCL-fusion-" + NewUniqueId())
                             .Op("_nccl_logical_fusion");
  for (const OpNode* nccl_op : nccl_ops) {
    fusion_builder = fusion_builder.Input(
        "in", GenLogicalBlobName(nccl_op->op().BnInOp2Lbi(nccl_op->op().SoleIbn())));
    // TODO
  }

  return Maybe<void>::Ok();
}

Maybe<void> NcclLogicalOpFusionPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashMap<const OpNode*, int64_t> op_node2nccl_depth;
  HashMap<int64_t, std::vector<const OpNode*>> nccl_depth2nccl_ops;
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    int64_t nccl_depth = 0;
    op_graph.ForEachDataAndCtrlInNode(node, [&](const OpNode* in_node) {
      auto it = op_node2nccl_depth.find(in_node);
      CHECK(it != op_node2nccl_depth.end());  // topo search
      nccl_depth = std::max(nccl_depth, it->second);
    });
    if (IsNcclLogicalOpNode(node)) {
      nccl_depth++;  // ONLY nccl node update depth
      nccl_depth2nccl_ops[nccl_depth].push_back(node);
    }
    CHECK(op_node2nccl_depth.emplace(node, nccl_depth).second);
  });

  if (nccl_depth2nccl_ops.empty()) { return Maybe<void>::Ok(); }

  for (const auto& pair : nccl_depth2nccl_ops) {
    HashMap<int64_t, HashMap<Shape, std::vector<const OpNode*>>> chain2hierarchy2nccl_ops;
    for (const OpNode* nccl_op : pair.second) {
      int64_t logical_chain_id = nccl_op->op().op_conf().logical_chain_id();
      const auto& hierarchy = nccl_op->parallel_desc().hierarchy();
      chain2hierarchy2nccl_ops[logical_chain_id][hierarchy].push_back(nccl_op);
    }
    for (const auto& chain_pair : chain2hierarchy2nccl_ops) {
      for (const auto& hierarchy_pair : chain_pair.second) {
        JUST(ReplaceNcclOpsWithFusionOp(job_builder, hierarchy_pair.second));
      }
    }
  }

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("NcclLogicalOpFusionPass", NcclLogicalOpFusionPass);

}  // namespace oneflow

#endif  // WITH_CUDA
