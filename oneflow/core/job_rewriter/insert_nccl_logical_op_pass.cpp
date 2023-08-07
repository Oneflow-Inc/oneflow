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
#include "oneflow/core/auto_parallel/auto_memory.h"
#include "oneflow/core/common/util.h"
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

DEFINE_ENV_INTEGER(ONEFLOW_GRAPH_MAX_NCCL_COMPUTE_STREAM, 8);

namespace {

class InsertNcclLogicalOpPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InsertNcclLogicalOpPass);
  InsertNcclLogicalOpPass() = default;
  ~InsertNcclLogicalOpPass() = default;

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

const std::string kNcclLogicalOpNamePrefix = "System-NCCL-Logical";

bool IsTickOpConf(const OperatorConf& op_conf) {
  if (IsClassRegistered<int32_t, IsTickTockOpTypeCase>(op_conf.op_type_case())) { return true; }
  if (op_conf.has_user_conf()) {
    const std::string& user_type_name = op_conf.user_conf().op_type_name();
    if (user_type_name == "cast_to_tick" || user_type_name == "acc_ctrl_tick") { return true; }
  }
  return false;
}

bool IsBreakpointOpNode(const OpNode* node) {
  // NOTE(chengcheng): breakpoint op is special which CANNOT through subgraph such as:
  //   variable, tick, repeat/acc/pack/unpack change timeshape
  const Operator& op = node->op();
  const OperatorConf& op_conf = op.op_conf();
  // TODO(chengcheng): filter ops which has special type
  // TODO(chengcheng): get stream by op type
  if (op_conf.has_variable_conf()                                                   /* varialbe */
      || IsTickOpConf(op_conf)                                                      /* tick */
      || op_conf.has_input_conf() || op_conf.has_output_conf()                      /* io */
      || op_conf.has_wait_and_send_ids_conf() || op_conf.has_callback_notify_conf() /* ctrl */
      || op_conf.has_image_decoder_random_crop_resize_conf() /* gpu decode */) {
    return true;
  }

  if (op_conf.has_user_conf()) {
    const std::string& user_type_name = op_conf.user_conf().op_type_name();
    if (user_type_name == "repeat" || user_type_name == "pack" || user_type_name == "unpack"
        || user_type_name == "identity_buffer") {
      return true;
    }
    if (!EnableLogicalChain()) {
      // NOTE(chengcheng): in old task graph chain version, consider acc as breakpoint node
      if (user_type_name == "acc") { return true; }
    }
  }
  return false;
}

bool IsAccOpNode(const OpNode* node) {
  return node->op().op_conf().has_user_conf()
         && node->op().op_conf().user_conf().op_type_name() == "acc";
}

bool IsRepeatOpNode(const OpNode* node) {
  return node->op().op_conf().has_user_conf()
         && node->op().op_conf().user_conf().op_type_name() == "repeat";
}

std::shared_ptr<const Shape> GetOpNodeTimeShape(const OpNode* op_node) {
  return CHECK_JUST(op_node->op().GetOpTimeShape());
}

std::shared_ptr<const Shape> GetOpNodeInputTimeShape(const OpNode* op_node) {
  return CHECK_JUST(op_node->op().GetInputBlobFastestTimeShape());
}

std::shared_ptr<const Shape> GetOpNodeFastestTimeShape(const OpNode* op_node) {
  return CHECK_JUST(op_node->op().GetInputOutputFastestTimeShape());
}

bool SharedPtrShapeEqual(const std::shared_ptr<const Shape>& lhs,
                         const std::shared_ptr<const Shape>& rhs) {
  return (*lhs) == (*rhs);
}

void FindAllConnectedSubgraphForGpuExecOrder(std::vector<HashSet<const OpNode*>>* ret,
                                             const OpGraph& op_graph,
                                             const std::vector<const OpNode*>& order) {
  // NOTE(chengcheng): acc subgraph may greater than fw/bw subgraph. we need use max time shape.
  std::shared_ptr<const Shape> seed_time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  op_graph.ForEachNode([&](const OpNode* node) {
    std::shared_ptr<const Shape> this_time_shape = GetOpNodeFastestTimeShape(node);
    if (this_time_shape->elem_cnt() > seed_time_shape->elem_cnt()) {
      seed_time_shape = this_time_shape;
    }
  });

  VLOG(2) << " seed time shape = " << seed_time_shape->ToString();

  HashSet<const OpNode*> visited;

  for (const OpNode* seed_node : order) {
    if (visited.find(seed_node) != visited.end()) { continue; }
    CHECK(visited.insert(seed_node).second);
    const ParallelDesc& seed_parallel_desc = seed_node->parallel_desc();
    // NOTE(chengcheng): ONLY consider GPU op and parallel num > 1.
    if (seed_parallel_desc.device_type() != DeviceType::kCUDA) { continue; }
    if (seed_parallel_desc.parallel_num() <= 1) { continue; }
    // NOTE(chengcheng): using fastest time shape for merge acc into bw subgraph.
    if (!SharedPtrShapeEqual(GetOpNodeFastestTimeShape(seed_node), seed_time_shape)) { continue; }
    if (IsBreakpointOpNode(seed_node)) { continue; }
    // NOTE(chengcheng):
    //   stream name hint maybe set by other job pass like replace embedding.
    //   we cannot replace stream name in subgraph
    if (seed_node->op().op_conf().has_stream_name_hint()) { continue; }

    HashSet<const OpNode*> this_subgraph;
    std::queue<const OpNode*> queued_nodes;

    queued_nodes.push(seed_node);
    while (!queued_nodes.empty()) {
      const OpNode* cur_node = queued_nodes.front();
      queued_nodes.pop();

      CHECK(cur_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc));
      CHECK(this_subgraph.insert(cur_node).second);

      cur_node->ForEachNodeOnInOutEdge([&](const OpNode* next_node) {
        if (visited.find(next_node) == visited.end() && (!IsBreakpointOpNode(next_node))
            && next_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc)
            && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(next_node), seed_time_shape)) {
          CHECK(visited.insert(next_node).second);
          queued_nodes.push(next_node);
        }
      });
    }

    if (this_subgraph.size() > 1) {
      ret->emplace_back(HashSet<const OpNode*>());
      ret->back().swap(this_subgraph);
    }
  }

  std::sort(ret->begin(), ret->end(),
            [](const HashSet<const OpNode*>& lhs, const HashSet<const OpNode*>& rhs) {
              return lhs.size() > rhs.size();
            });
}

bool TryBuildNcclBy1DHierarchy(OperatorConf* ret, const SbpParallel& src_sbp,
                               const SbpParallel& dst_sbp, const std::string& lbn,
                               const int64_t scope_symbol_id, const BlobDesc& logical_blob_desc,
                               const int64_t parallel_num) {
  auto CanSplitAtDim = [&](int64_t dim) -> bool {
    if (logical_blob_desc.shape().NumAxes() <= dim) { return false; }
    return logical_blob_desc.shape().At(dim) % parallel_num == 0;
  };
  if (src_sbp.has_partial_sum_parallel() && dst_sbp.has_broadcast_parallel()) {
    // P->B : AllReduce
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2B-" + NewUniqueId())
               .Op("_nccl_logical_all_reduce")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (CanSplitAtDim(0)
             && (src_sbp.has_partial_sum_parallel() && dst_sbp.has_split_parallel())
             && (dst_sbp.split_parallel().axis() == 0)) {
    // P->S(0) : ReduceScatter
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2S-" + NewUniqueId())
               .Op("_nccl_logical_reduce_scatter")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (CanSplitAtDim(0) && (src_sbp.has_split_parallel() && dst_sbp.has_broadcast_parallel())
             && (src_sbp.split_parallel().axis() == 0)) {
    // S(0)->B : AllGather
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2B-" + NewUniqueId())
               .Op("_nccl_logical_all_gather")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (src_sbp.has_split_parallel() && dst_sbp.has_broadcast_parallel()
             && src_sbp.split_parallel().axis() > 0
             && CanSplitAtDim(src_sbp.split_parallel().axis())) {
    // S(1)->B : AllGather Noncontinuous
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2B-" + NewUniqueId())
               .Op("_nccl_logical_all_gather_noncontinuous")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (src_sbp.has_split_parallel() && dst_sbp.has_split_parallel()
             && src_sbp.split_parallel().axis() != dst_sbp.split_parallel().axis()
             && CanSplitAtDim(src_sbp.split_parallel().axis())
             && CanSplitAtDim(dst_sbp.split_parallel().axis())) {
    // S(in)->S(out) : All2All
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-S2S-" + NewUniqueId())
               .Op("_nccl_logical_s2s")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (CanSplitAtDim(dst_sbp.split_parallel().axis())
             && (src_sbp.has_partial_sum_parallel() && dst_sbp.has_split_parallel())
             && (dst_sbp.split_parallel().axis() > 0)) {
    // P->S(1) : ReduceScatter Noncontinuous
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-P2S-" + NewUniqueId())
               .Op("_nccl_logical_reduce_scatter_noncontinuous")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  } else if (!dst_sbp.has_partial_sum_parallel()) {
    *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(Send)2(Recv)-"
                                             + NewUniqueId())
               .Op("_nccl_logical_send_recv")
               .Input("in", lbn)
               .Output("out")
               .Attr<std::vector<std::string>>("src_reduced_nd_sbp", {SbpToString(src_sbp)})
               .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", {SbpToString(dst_sbp)})
               .ScopeSymbolId(scope_symbol_id)
               .Build()
               .op_conf();
    return true;
  }
  return false;
}

bool TryBuildNcclBy2DHierarchySameDim0(OperatorConf* ret, const NdSbp& src_nd_sbp,
                                       const NdSbp& dst_nd_sbp,
                                       const std::shared_ptr<Shape>& hierarchy,
                                       const std::string& lbn, const int64_t scope_symbol_id,
                                       const BlobDesc& logical_blob_desc) {
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
  CHECK(src_nd_sbp.sbp_parallel(0) == dst_nd_sbp.sbp_parallel(0));
  const SbpParallel& src_dim1_sbp = src_nd_sbp.sbp_parallel(1);
  const SbpParallel& dst_dim1_sbp = dst_nd_sbp.sbp_parallel(1);

  // split when dim0 sbp is split parallel
  DimVector dim_vec = logical_blob_desc.shape().dim_vec();
  if (src_nd_sbp.sbp_parallel(0).has_split_parallel()) {
    const int64_t axis = src_nd_sbp.sbp_parallel(0).split_parallel().axis();
    dim_vec.at(axis) /= hierarchy->At(0);
  }
  const int64_t num_ranks = hierarchy->At(1);

  if (src_dim1_sbp.has_partial_sum_parallel() && dst_dim1_sbp.has_broadcast_parallel()) {
    // (*, P)->(*, B) : AllReduce
    *ret =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(*P)2(*B)-" + NewUniqueId())
            .Op("_nccl_logical_2D_same_dim0_all_reduce")
            .Input("in", lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
            .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
            .ScopeSymbolId(scope_symbol_id)
            .Build()
            .op_conf();
    return true;
  } else if ((src_dim1_sbp.has_split_parallel() && dst_dim1_sbp.has_broadcast_parallel())
             && (src_dim1_sbp.split_parallel().axis() == 0) && (dim_vec.at(0) % num_ranks == 0)) {
    // (*, S(0)) -> (*, B) : AllGather
    *ret =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(*S0)2(*B)-" + NewUniqueId())
            .Op("_nccl_logical_2D_same_dim0_all_gather")
            .Input("in", lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
            .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
            .ScopeSymbolId(scope_symbol_id)
            .Build()
            .op_conf();
    return true;
  } else if (src_dim1_sbp.has_split_parallel() && dst_dim1_sbp.has_broadcast_parallel()
             && (src_dim1_sbp.split_parallel().axis() > 0)
             && (dim_vec.at(src_dim1_sbp.split_parallel().axis()) % num_ranks == 0)) {
    // (*, S(1)) -> (*, B) : AllGather Noncontinuous
    *ret =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(*S1)2(*B)-" + NewUniqueId())
            .Op("_nccl_logical_2D_same_dim0_all_gather_noncontinuous")
            .Input("in", lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
            .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
            .ScopeSymbolId(scope_symbol_id)
            .Build()
            .op_conf();
    return true;
  } else if ((src_dim1_sbp.has_split_parallel() && dst_dim1_sbp.has_split_parallel())
             && (src_dim1_sbp.split_parallel().axis() != dst_dim1_sbp.split_parallel().axis())
             && (dim_vec.at(src_dim1_sbp.split_parallel().axis()) % num_ranks == 0)
             && (dim_vec.at(dst_dim1_sbp.split_parallel().axis()) % num_ranks == 0)) {
    // (*, S(src_split_axis)) -> (*, S(dst_split_axis)) : All2All
    *ret =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(*S)2(*S)-" + NewUniqueId())
            .Op("_nccl_logical_2D_same_dim0_all2all")
            .Input("in", lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
            .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
            .ScopeSymbolId(scope_symbol_id)
            .Build()
            .op_conf();
    return true;
  }
  return false;
}

bool TryBuildNcclBy2DHierarchySameDim1(OperatorConf* ret, const NdSbp& src_nd_sbp,
                                       const NdSbp& dst_nd_sbp,
                                       const std::shared_ptr<Shape>& hierarchy,
                                       const std::string& lbn, const int64_t scope_symbol_id,
                                       const BlobDesc& logical_blob_desc) {
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
  CHECK(src_nd_sbp.sbp_parallel(1) == dst_nd_sbp.sbp_parallel(1));
  const SbpParallel& src_dim1_sbp = src_nd_sbp.sbp_parallel(0);
  const SbpParallel& dst_dim1_sbp = dst_nd_sbp.sbp_parallel(0);
  if (src_dim1_sbp.has_partial_sum_parallel() && dst_dim1_sbp.has_broadcast_parallel()) {
    // (P, *) -> (B, *) : AllReduce
    *ret =
        user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(P*)2(B*)-" + NewUniqueId())
            .Op("_nccl_logical_2D_same_dim1_all_reduce")
            .Input("in", lbn)
            .Output("out")
            .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
            .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
            .ScopeSymbolId(scope_symbol_id)
            .Build()
            .op_conf();
    return true;
  }
  return false;
}

bool TryBuildNcclBy2DHierarchyOthers(OperatorConf* ret, const NdSbp& src_nd_sbp,
                                     const NdSbp& dst_nd_sbp,
                                     const std::shared_ptr<Shape>& hierarchy,
                                     const std::string& lbn, const int64_t scope_symbol_id,
                                     const BlobDesc& logical_blob_desc) {
  CHECK_EQ(src_nd_sbp.sbp_parallel_size(), 2);
  CHECK_EQ(dst_nd_sbp.sbp_parallel_size(), 2);
  // send recv is dealing with same 0-Dim
  VLOG_IF(3, src_nd_sbp.sbp_parallel(0) == dst_nd_sbp.sbp_parallel(0))
      << "send recv is dealing with same 0-Dim, src sbp " << NdSbpToString(src_nd_sbp)
      << ", dst sbp " << NdSbpToString(dst_nd_sbp);
  // send recv is dealing with same 1-Dim, such as (B, S0) -> (S0, S0)
  VLOG_IF(3, ((src_nd_sbp.sbp_parallel(1) == dst_nd_sbp.sbp_parallel(1))
              && !(NdSbpAllSameSplitParallel(src_nd_sbp) || NdSbpAllSameSplitParallel(dst_nd_sbp))))
      << "send recv is dealing with same 1-Dim,  src sbp " << NdSbpToString(src_nd_sbp)
      << ", dst sbp " << NdSbpToString(dst_nd_sbp);
  // send recv can not dealing with P in dst_nd_sbp
  if (NdSbpHasPartialParallel(dst_nd_sbp)) return false;
  *ret = user_op::UserOpConfWrapperBuilder(kNcclLogicalOpNamePrefix + "-(Send)2(Recv)-"
                                           + NewUniqueId())
             .Op("_nccl_logical_send_recv")
             .Input("in", lbn)
             .Output("out")
             .Attr<std::vector<std::string>>("src_reduced_nd_sbp", NdSbpToStringList(src_nd_sbp))
             .Attr<std::vector<std::string>>("dst_reduced_nd_sbp", NdSbpToStringList(dst_nd_sbp))
             .ScopeSymbolId(scope_symbol_id)
             .Build()
             .op_conf();
  return true;
}

Maybe<int64_t> BuildScopeWithReducedParallelDesc(int64_t old_scope_symbol_id,
                                                 const ParallelDesc& parallel_desc) {
  auto* scope_storage = Singleton<symbol::Storage<Scope>>::Get();
  CHECK_OR_RETURN(scope_storage->Has(old_scope_symbol_id));
  auto old_scope = scope_storage->GetPtr(old_scope_symbol_id);
  std::shared_ptr<Scope> new_scope;
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    new_scope =
        JUST(builder->BuildScopeWithNewParallelConf(old_scope, parallel_desc.parallel_conf()));
    return Maybe<void>::Ok();
  }));
  // NOTE(chengcheng): need sync vm for get scope right now
  JUST(vm::CurrentRankSync());
  CHECK_OR_RETURN(new_scope);
  return JUST(new_scope->symbol_id());
}

bool TryBuildNcclLogicalOpConf(OperatorConf* ret, const OpNode* src_node, const OpNode* dst_node,
                               const LogicalBlobId& lbi, ParallelDesc* src_reduced_parallel_desc,
                               ParallelDesc* dst_reduced_parallel_desc, NdSbp* src_reduced_nd_sbp,
                               NdSbp* dst_reduced_nd_sbp) {
  if (!src_node->op().op_conf().has_scope_symbol_id()) { return false; /* device_tick */ }
  const std::string lbn = GenLogicalBlobName(lbi);
  const BlobDesc& logical_blob_desc = src_node->LogicalBlobDesc4Lbi(lbi);

  // reduce hierarchy
  InOutParallelDimReduce(src_node->parallel_desc(), dst_node->parallel_desc(),
                         src_node->NdSbp4Lbi(lbi), dst_node->NdSbp4Lbi(lbi),
                         src_reduced_parallel_desc, dst_reduced_parallel_desc, src_reduced_nd_sbp,
                         dst_reduced_nd_sbp, logical_blob_desc.shape());

  CHECK_EQ(src_reduced_parallel_desc->parallel_num(), dst_reduced_parallel_desc->parallel_num());
  std::shared_ptr<Shape> src_reduced_hierarchy = src_reduced_parallel_desc->hierarchy();
  std::shared_ptr<Shape> dst_reduced_hierarchy = dst_reduced_parallel_desc->hierarchy();

  if ((*src_reduced_hierarchy) == (*dst_reduced_hierarchy)
      && (*src_reduced_nd_sbp) == (*dst_reduced_nd_sbp)) {
    // one to one
    return false;
  }

  // NOTE(chengcheng): nccl donot support dynamic shape.
  if (logical_blob_desc.is_dynamic()) { return false; }
  CHECK_GT(logical_blob_desc.shape().elem_cnt(), 0)
      << dst_node->op().op_name() << " consume " << GenLogicalBlobName(lbi) << ", "
      << *CHECK_JUST(PlacementToString(*src_reduced_parallel_desc)) << " "
      << NdSbpToString(*src_reduced_nd_sbp) << " -> "
      << *CHECK_JUST(PlacementToString(*dst_reduced_parallel_desc)) << " "
      << NdSbpToString(*dst_reduced_nd_sbp);

  int64_t scope_symbol_id = CHECK_JUST(BuildScopeWithReducedParallelDesc(
      src_node->op().op_conf().scope_symbol_id(), *src_reduced_parallel_desc));

  if (src_reduced_hierarchy->NumAxes() == 1 && dst_reduced_hierarchy->NumAxes() == 1) {
    return TryBuildNcclBy1DHierarchy(ret, src_reduced_nd_sbp->sbp_parallel(0),
                                     dst_reduced_nd_sbp->sbp_parallel(0), lbn, scope_symbol_id,
                                     logical_blob_desc, src_reduced_parallel_desc->parallel_num());
  } else if (src_reduced_hierarchy->NumAxes() == 2
             && (*src_reduced_hierarchy == *dst_reduced_hierarchy)) {
    bool got_nccl = false;
    if (src_reduced_nd_sbp->sbp_parallel(0) == dst_reduced_nd_sbp->sbp_parallel(0)) {
      // TODO(): same dim 0 need to deal with (*, P) -> (*, S)
      got_nccl = TryBuildNcclBy2DHierarchySameDim0(ret, *src_reduced_nd_sbp, *dst_reduced_nd_sbp,
                                                   src_reduced_hierarchy, lbn, scope_symbol_id,
                                                   logical_blob_desc);
    } else if (src_reduced_nd_sbp->sbp_parallel(1) == dst_reduced_nd_sbp->sbp_parallel(1)) {
      if (!(NdSbpAllSameSplitParallel(*src_reduced_nd_sbp)
            || NdSbpAllSameSplitParallel(*dst_reduced_nd_sbp))) {
        got_nccl = TryBuildNcclBy2DHierarchySameDim1(ret, *src_reduced_nd_sbp, *dst_reduced_nd_sbp,
                                                     src_reduced_hierarchy, lbn, scope_symbol_id,
                                                     logical_blob_desc);
      }
    }
    if (!got_nccl) {
      got_nccl = TryBuildNcclBy2DHierarchyOthers(ret, *src_reduced_nd_sbp, *dst_reduced_nd_sbp,
                                                 src_reduced_hierarchy, lbn, scope_symbol_id,
                                                 logical_blob_desc);
    }
    VLOG_IF(3, !got_nccl) << "Cannot get nccl logical op for 2D sbp, src nd sbp "
                          << NdSbpToString(*src_reduced_nd_sbp) << ", dst nd sbp "
                          << NdSbpToString(*dst_reduced_nd_sbp) << ".";
    return got_nccl;
  }
  return false;
}

void InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(
    HashMap<std::string, OperatorConf>* subgraph_op_name2conf, HashSet<std::string>* mut_op_names,
    std::vector<OperatorConf>* nccl_op_confs, std::vector<ParallelConf>* nccl_op_parallel_confs,
    const std::vector<const OpNode*>& subgraph_ordered_nodes,
    const HashMap<const OpNode*, int64_t>& node2subgraph_order) {
  for (const OpNode* dst_node : subgraph_ordered_nodes) {
    const std::string& dst_op_name = dst_node->op().op_name();
    for (const OpEdge* op_edge : dst_node->in_edges()) {
      const OpNode* src_node = op_edge->src_node();
      const std::string& src_op_name = src_node->op().op_name();
      CHECK(src_node != dst_node);
      if (src_node->parallel_desc().EqualsIgnoringHierarchy(dst_node->parallel_desc())) {
        // NOTE(chengcheng): We don't care src node whether in this subgraph, or whether is repeat
        //  op, or whether is breaking op. We ONLY care src node is same placement with dst.
        //  So, we can handle both ZeRO from variable and in GradAcc from repeat and in Pipeline.
        for (const LogicalBlobId& lbi : op_edge->lbis()) {
          OperatorConf nccl_op;
          ParallelDesc src_reduced_parallel_desc = op_edge->src_node()->parallel_desc();
          ParallelDesc dst_reduced_parallel_desc = op_edge->dst_node()->parallel_desc();
          NdSbp src_reduced_nd_sbp;
          NdSbp dst_reduced_nd_sbp;
          if (!TryBuildNcclLogicalOpConf(&nccl_op, src_node, dst_node, lbi,
                                         &src_reduced_parallel_desc, &dst_reduced_parallel_desc,
                                         &src_reduced_nd_sbp, &dst_reduced_nd_sbp)) {
            continue;
          }
          mut_op_names->insert(dst_op_name);
          // insert nccl op
          user_op::UserOpConfWrapper nccl_op_wrapper(nccl_op);
          for (const std::string& ibn : op_edge->lbi2ibns().at(lbi)) {
            std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
                &subgraph_op_name2conf->at(dst_op_name), ibn, nccl_op_wrapper.output("out", 0));
            CHECK(old_lbn == GenLogicalBlobName(lbi));
          }

          // NOTE(chengcheng): Do NOT add ctrl edge for nccl fusion.
          nccl_op_confs->emplace_back(nccl_op);
          // NOTE(chengcheng, guoran): set nccl op as dst_node parallel_conf (hierarchy) may check
          //   failed in complier, so need use dst_node reduced_parallel_conf.
          nccl_op_parallel_confs->emplace_back(dst_reduced_parallel_desc.parallel_conf());
          VLOG(2) << " insert nccl op: " << nccl_op.name() << " from [" << src_op_name << "] to ["
                  << dst_op_name << "]\n";
        }
      }
    }
  }
}

void GenAfterAccSubgraph(std::vector<const OpNode*>* ordered_after_acc_subgraph,
                         const HashMap<const OpNode*, int64_t>& op_node2global_order,
                         const std::vector<const OpNode*>& ordered_acc_op_nodes) {
  std::shared_ptr<const Shape> seed_time_shape = std::make_shared<const Shape>(Shape({1, 1}));
  const ParallelDesc& seed_parallel_desc = ordered_acc_op_nodes.front()->parallel_desc();
  HashSet<const OpNode*> visited;
  std::queue<const OpNode*> queued_nodes;
  auto SearchToNextNode = [&](const OpNode* cur_node, const OpNode* next_node, const OpEdge* edge) {
    if (visited.find(next_node) == visited.end() && (!IsBreakpointOpNode(next_node))
        && next_node->parallel_desc().EqualsIgnoringHierarchy(seed_parallel_desc)
        && SharedPtrShapeEqual(GetOpNodeFastestTimeShape(next_node), seed_time_shape)) {
      CHECK(visited.insert(next_node).second);
      queued_nodes.push(next_node);
    }
  };

  auto CmpOpNodeOrder = [&](const OpNode* lhs, const OpNode* rhs) {
    return op_node2global_order.at(lhs) < op_node2global_order.at(rhs);
  };

  for (const OpNode* acc_node : ordered_acc_op_nodes) {
    for (const OpEdge* out_edge : acc_node->out_edges()) {
      const OpNode* seed_node = out_edge->dst_node();
      SearchToNextNode(acc_node, seed_node, out_edge);
    }
  }

  while (!queued_nodes.empty()) {
    const OpNode* cur_node = queued_nodes.front();
    queued_nodes.pop();

    ordered_after_acc_subgraph->push_back(cur_node);

    for (const OpEdge* in_edge : cur_node->in_edges()) {
      SearchToNextNode(cur_node, in_edge->src_node(), in_edge);
    }
    for (const OpEdge* out_edge : cur_node->out_edges()) {
      SearchToNextNode(cur_node, out_edge->dst_node(), out_edge);
    }
  }

  std::sort(ordered_after_acc_subgraph->begin(), ordered_after_acc_subgraph->end(), CmpOpNodeOrder);
}

struct InsertNcclSubGraph {
  std::vector<const OpNode*> ordered_op_nodes;
  int64_t begin_op_global_order;
  int64_t end_op_global_order;
  const OpNode* begin_op;
  const OpNode* end_op;
};

struct PlacementNcclSubGraghsInfo {
  std::vector<std::shared_ptr<InsertNcclSubGraph>> ordered_subgraph;
  std::vector<const OpNode*> ordered_acc_op_nodes;
  const ParallelDesc* seed_parallel_desc;
};

void InitInsertNcclSubGraphInfoFromSet(
    std::shared_ptr<InsertNcclSubGraph> nccl_subgraph_info, const HashSet<const OpNode*>& subgraph,
    const HashMap<const OpNode*, int64_t>& op_node2global_order,
    const std::function<bool(const OpNode*, const OpNode*)>& CmpOpNodeOrder) {
  auto* subgraph_ordered_nodes = &nccl_subgraph_info->ordered_op_nodes;
  subgraph_ordered_nodes->assign(subgraph.begin(), subgraph.end());
  std::sort(subgraph_ordered_nodes->begin(), subgraph_ordered_nodes->end(), CmpOpNodeOrder);
  nccl_subgraph_info->begin_op = subgraph_ordered_nodes->front();
  nccl_subgraph_info->end_op = subgraph_ordered_nodes->back();
  nccl_subgraph_info->begin_op_global_order = op_node2global_order.at(nccl_subgraph_info->begin_op);
  nccl_subgraph_info->end_op_global_order = op_node2global_order.at(nccl_subgraph_info->end_op);
  CHECK(nccl_subgraph_info->begin_op != nccl_subgraph_info->end_op);
  CHECK_LT(nccl_subgraph_info->begin_op_global_order, nccl_subgraph_info->end_op_global_order);
}

std::string GetStreamIndexName(uint32_t id) { return "NCCL_COMPUTE_" + std::to_string(id); }

int64_t InsertNcclLogicalOpsInSubGraph(const OpGraph& op_graph, JobBuilder* job_builder,
                                       const std::vector<const OpNode*>& subgraph_ordered_nodes,
                                       int64_t* nccl_compute_stream_id,
                                       const int64_t logical_chain_id) {
  HashMap<const OpNode*, int64_t> node2subgraph_order;
  node2subgraph_order.reserve(subgraph_ordered_nodes.size());
  for (int64_t i = 0; i < subgraph_ordered_nodes.size(); ++i) {
    CHECK(node2subgraph_order.emplace(subgraph_ordered_nodes.at(i), i).second);
  }

  VLOG(3) << " ======================================================================== \n"
          << " Try insert nccl logical ops into Graph: " << job_builder->job().job_conf().job_name()
          << " , logical_chain: " << logical_chain_id << ". Begin...\n";

  HashSet<std::string> mut_op_names;
  HashMap<std::string, OperatorConf> subgraph_op_name2conf;
  for (const OpNode* this_node : subgraph_ordered_nodes) {
    VLOG(3) << "logical_chain: " << logical_chain_id << " , op: " << this_node->op().op_name();
    CHECK(
        subgraph_op_name2conf.emplace(this_node->op().op_name(), this_node->op().op_conf()).second);
  }

  std::vector<OperatorConf> nccl_op_confs;
  std::vector<ParallelConf> nccl_op_parallel_confs;
  // NOTE(chengcheng): ONLY support insert nccl to dst for memory.
  InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(&subgraph_op_name2conf, &mut_op_names,
                                                 &nccl_op_confs, &nccl_op_parallel_confs,
                                                 subgraph_ordered_nodes, node2subgraph_order);

  VLOG(3) << " ======================================================================== \n"
          << " Try insert nccl logical ops into Graph: " << job_builder->job().job_conf().job_name()
          << " , logical_chain: " << logical_chain_id << ". End.\n";

  // NOTE(chengcheng): For NCCL logical correct exec order in pipeline multi-subgraph.
  if (nccl_op_confs.empty()) { return 0; }
  const int64_t max_nccl_stream_count = EnvInteger<ONEFLOW_GRAPH_MAX_NCCL_COMPUTE_STREAM>();
  if ((*nccl_compute_stream_id) >= max_nccl_stream_count) {
    return 0;  // NOTE(chengcheng): ONLY support kMaxNcclComputeStreamCount insert nccl subgraphs.
  }
  std::string stream_index_name = GetStreamIndexName(*nccl_compute_stream_id);
  // NOTE(chengcheng): ONLY valid subgraph will increase nccl stream id.
  (*nccl_compute_stream_id)++;

  // NOTE(chengcheng): set ALL subgraph op and ALL nccl op stream index and logical chain id.
  for (auto& pair : subgraph_op_name2conf) {
    mut_op_names.insert(pair.first);
    pair.second.set_stream_name_hint(stream_index_name);
    pair.second.set_logical_chain_id(logical_chain_id);
  }
  for (auto& nccl_op : nccl_op_confs) {
    nccl_op.set_stream_name_hint(stream_index_name);
    nccl_op.set_logical_chain_id(logical_chain_id);
  }

  std::vector<OperatorConf> mut_op_confs;
  mut_op_confs.reserve(mut_op_names.size());
  for (const std::string& mut_op_name : mut_op_names) {
    mut_op_confs.emplace_back(subgraph_op_name2conf.at(mut_op_name));
  }
  job_builder->MutOpsOnlyOnce(mut_op_confs);

  CHECK_EQ(nccl_op_confs.size(), nccl_op_parallel_confs.size());
  for (int64_t i = 0; i < nccl_op_confs.size(); ++i) {
    CHECK_JUST(job_builder->AddOp(nccl_op_parallel_confs.at(i), nccl_op_confs.at(i)));
  }
  VLOG(3) << " In logical chain id: " << logical_chain_id
          << " insert nccl op num = " << nccl_op_confs.size()
          << " and origin chain op num = " << subgraph_ordered_nodes.size();
  return nccl_op_confs.size() + subgraph_ordered_nodes.size();
}

void InsertNcclLogicalOpsAfterAcc(const OpGraph& op_graph, JobBuilder* job_builder,
                                  const std::vector<const OpNode*>& ordered_acc_op_nodes,
                                  const HashMap<const OpNode*, int64_t>& op_node2global_order,
                                  const int64_t nccl_compute_stream_id,
                                  const int64_t logical_chain_id) {
  // insert nccl ops after acc
  std::vector<const OpNode*> ordered_after_acc_subgraph;
  GenAfterAccSubgraph(&ordered_after_acc_subgraph, op_node2global_order, ordered_acc_op_nodes);
  if (ordered_after_acc_subgraph.size() <= 1) { return; }

  HashMap<const OpNode*, int64_t> node2subgraph_order;
  node2subgraph_order.reserve(ordered_after_acc_subgraph.size());
  for (int64_t i = 0; i < ordered_after_acc_subgraph.size(); ++i) {
    CHECK(node2subgraph_order.emplace(ordered_after_acc_subgraph.at(i), i).second);
  }

  std::vector<OperatorConf> after_acc_nccl_op_confs;
  std::vector<ParallelConf> after_acc_nccl_parallel_confs;
  HashSet<std::string> mut_op_names;
  HashMap<std::string, OperatorConf> acc_subgraph_op_name2conf;
  for (const OpNode* this_node : ordered_after_acc_subgraph) {
    CHECK(acc_subgraph_op_name2conf.emplace(this_node->op().op_name(), this_node->op().op_conf())
              .second);
    VLOG(3) << "After Acc logical_chain: " << logical_chain_id
            << " , op: " << this_node->op().op_name();
  }

  InsertNcclLogicalOpsAsCloseAsPossibleToDstNode(
      &acc_subgraph_op_name2conf, &mut_op_names, &after_acc_nccl_op_confs,
      &after_acc_nccl_parallel_confs, ordered_after_acc_subgraph, node2subgraph_order);

  if (after_acc_nccl_op_confs.empty()) {
    CHECK(after_acc_nccl_parallel_confs.empty());
    CHECK(mut_op_names.empty());
  } else {
    std::string stream_index_name = GetStreamIndexName(nccl_compute_stream_id);

    // set logical chain id and stream name for ops after acc
    for (auto& pair : acc_subgraph_op_name2conf) {
      mut_op_names.insert(pair.first);
      pair.second.set_stream_name_hint(stream_index_name);
      pair.second.set_logical_chain_id(logical_chain_id);
    }
    for (auto& nccl_op : after_acc_nccl_op_confs) {
      nccl_op.set_stream_name_hint(stream_index_name);
      nccl_op.set_logical_chain_id(logical_chain_id);
    }

    // insert nccl ops after acc
    std::vector<OperatorConf> mut_op_confs;
    mut_op_confs.reserve(mut_op_names.size());
    for (const std::string& mut_op_name : mut_op_names) {
      mut_op_confs.emplace_back(acc_subgraph_op_name2conf.at(mut_op_name));
    }
    job_builder->MutOpsOnlyOnce(mut_op_confs);

    CHECK_EQ(after_acc_nccl_op_confs.size(), after_acc_nccl_parallel_confs.size());
    for (int64_t i = 0; i < after_acc_nccl_op_confs.size(); ++i) {
      CHECK_JUST(
          job_builder->AddOp(after_acc_nccl_parallel_confs.at(i), after_acc_nccl_op_confs.at(i)));
    }
  }
}

Maybe<void> InsertNcclLogicalOpPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  std::vector<const OpNode*> ordered_op_nodes;
  if (ParseBooleanFromEnv("DISABLE_LOGICAL_STRAIGHTEN", false)) {
    op_graph.TopoForEachNodeWithCtrlEdge(
        [&](const OpNode* node) { ordered_op_nodes.emplace_back(node); });
  } else {
    auto_parallel::StraightenOpGraph(op_graph, &ordered_op_nodes);
  }

  HashMap<const OpNode*, int64_t> op_node2global_order;
  for (int32_t global_order = 0; global_order < ordered_op_nodes.size(); global_order++) {
    op_node2global_order.emplace(ordered_op_nodes[global_order], global_order);
  }

  std::vector<HashSet<const OpNode*>> subgraph_list;
  FindAllConnectedSubgraphForGpuExecOrder(&subgraph_list, op_graph, ordered_op_nodes);
  if (subgraph_list.size() == 0) { return Maybe<void>::Ok(); }

  // sign subgraph ops logical chain id for merge.
  int64_t global_logical_chain_id = 0;

  auto CmpOpNodeOrder = [&](const OpNode* lhs, const OpNode* rhs) {
    return op_node2global_order.at(lhs) < op_node2global_order.at(rhs);
  };
  auto CmpSubGraphOrder = [&](const std::shared_ptr<InsertNcclSubGraph>& lhs,
                              const std::shared_ptr<InsertNcclSubGraph>& rhs) {
    int64_t lhs_begin_op_global_order = op_node2global_order.at(lhs->ordered_op_nodes.front());
    int64_t rhs_begin_op_global_order = op_node2global_order.at(rhs->ordered_op_nodes.front());
    return lhs_begin_op_global_order < rhs_begin_op_global_order;
  };

  HashMap<std::string, PlacementNcclSubGraghsInfo> placement2subgraphs;
  for (const auto& subgraph : subgraph_list) {
    const OpNode* rand_node = *subgraph.begin();
    const ParallelDesc& this_parallel_desc = rand_node->parallel_desc();
    std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
    auto it = placement2subgraphs.find(key);
    if (it == placement2subgraphs.end()) {
      it = placement2subgraphs.emplace(key, PlacementNcclSubGraghsInfo()).first;
      it->second.seed_parallel_desc = &this_parallel_desc;
    } else {
      CHECK(this_parallel_desc.EqualsIgnoringHierarchy(*it->second.seed_parallel_desc));
    }
    auto& info = it->second;
    info.ordered_subgraph.emplace_back(std::make_shared<InsertNcclSubGraph>());
    InitInsertNcclSubGraphInfoFromSet(info.ordered_subgraph.back(), subgraph, op_node2global_order,
                                      CmpOpNodeOrder);
  }
  for (auto& pair : placement2subgraphs) {
    std::sort(pair.second.ordered_subgraph.begin(), pair.second.ordered_subgraph.end(),
              CmpSubGraphOrder);
  }

  for (const OpNode* this_node : ordered_op_nodes) {
    if (IsAccOpNode(this_node)) {
      const ParallelDesc& this_parallel_desc = this_node->parallel_desc();
      std::string key = GenParallelConfKey(this_parallel_desc.parallel_conf());
      auto it = placement2subgraphs.find(key);
      if (it != placement2subgraphs.end()) {
        it->second.ordered_acc_op_nodes.emplace_back(this_node);
      }
    }
  }

  for (auto& pair : placement2subgraphs) {
    PlacementNcclSubGraghsInfo& info = pair.second;

    // NOTE(chengcheng): insert nccl ops for each subgraph
    int64_t stream_offset = 0;
    int64_t total_op_num = 0;
    for (int i = 0; i < info.ordered_subgraph.size(); i++) {
      auto& ordered_op_nodes = info.ordered_subgraph.at(i)->ordered_op_nodes;
      int64_t this_op_num = InsertNcclLogicalOpsInSubGraph(
          op_graph, job_builder, ordered_op_nodes, &stream_offset, global_logical_chain_id++);
      total_op_num += this_op_num;
    }
    if (stream_offset >= 2 && total_op_num >= 1000) {
      LOG(WARNING) << " In Graph: " << job_builder->job().job_conf().job_name()
                   << " Placement: " << pair.first << " the total_op_num = " << total_op_num
                   << " and has " << stream_offset
                   << " different nccl stream which is possible to trigger cuda stream kernel "
                      "launch upper limit."
                   << " So the nccl logical kernel will from async to sync exec, which may affect "
                      "performance.";
      EagerCclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerCclCommMgr>::Get());
      comm_mgr->SetAsyncLaunchCclLogicalKernel(false);
    }

    // NOTE(chengcheng): insert acc for all subgraph with same placement group
    if (!info.ordered_acc_op_nodes.empty()) {
      InsertNcclLogicalOpsAfterAcc(op_graph, job_builder, info.ordered_acc_op_nodes,
                                   op_node2global_order, stream_offset++,
                                   global_logical_chain_id++);
    }
  }

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("InsertNcclLogicalOpPass", InsertNcclLogicalOpPass);

}  // namespace oneflow

#endif  // WITH_CUDA
