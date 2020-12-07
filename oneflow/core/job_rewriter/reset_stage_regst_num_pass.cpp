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
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/graph/compute_graph.h"
#include "oneflow/core/graph/stage_chain_graph.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace {

class ResetStageRegstNumPass final : public JobPass {
 public:
  ResetStageRegstNumPass(const ResetStageRegstNumPass&) = delete;
  ResetStageRegstNumPass(ResetStageRegstNumPass&&) = delete;
  ResetStageRegstNumPass() = default;
  ~ResetStageRegstNumPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    auto compute_graph = JUST(ComputeGraph::New(*job));
    auto stage_chain_graph = JUST(StageChainGraph::New(*compute_graph));
    stage_chain_graph->InitEdgeStatistics();
    {
      std::string job_name = ctx->job_desc().job_name();
      compute_graph->ToDotWithFilePath(std::string("compute_graph-") + job_name + ".dot");
      stage_chain_graph->ToDotWithFilePath(std::string("stage_chain_graph-") + job_name + ".dot");
    }
    JobBuilder job_builder(job);
    bool enable_stage_static_scheduling = ctx->job_desc().Bool("enable_stage_static_scheduling");
    return Apply(*compute_graph, *stage_chain_graph, &job_builder, enable_stage_static_scheduling);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_stage_buffer");
  }

  Maybe<void> Apply(const ComputeGraph& compute_graph, const StageChainGraph& stage_chain_graph,
                    JobBuilder* job_builder, bool enable_stage_static_scheduling) const {
    std::function<Maybe<int64_t>(const LogicalBlobId&)> BufferSize4Lbi;
    JUST(MakeGetterBufferSize4Lbi(compute_graph, &BufferSize4Lbi));
    HashMap<std::string, int64_t> op_name2each_output_regst_num;
    JUST(ForEachLbi7RequiredBufferSize(
        compute_graph, stage_chain_graph,
        [&](const LogicalBlobId& lbi, int64_t required_buffer_size) -> Maybe<void> {
          int64_t lbi_buffer_size = JUST(BufferSize4Lbi(lbi));
          if (required_buffer_size <= lbi_buffer_size) { return Maybe<void>::Ok(); }
          auto* buffer_size = &op_name2each_output_regst_num[lbi.op_name()];
          *buffer_size = std::max<int64_t>(*buffer_size, required_buffer_size);
          return Maybe<void>::Ok();
        }));
    for (auto& pair : op_name2each_output_regst_num) {
      JUST(ResetRegstNum(job_builder, pair.first, pair.second));
    }
    if (enable_stage_static_scheduling) {
      JUST(stage_chain_graph.MaybeForEachEdge([&](StageChainEdge* edge) -> Maybe<void> {
        size_t num_placement_ids = JUST(edge->NumStagePlacementInPath());
        if (!JUST(NeedStaticScheduling(edge, num_placement_ids))) { return Maybe<void>::Ok(); }
        JUST(AddCtrlFromSrcSourceToDstSource(job_builder, compute_graph, edge));
        return Maybe<void>::Ok();
      }));
      JUST(job_builder->MutCachedOpConfOnlyOnce());
    }
    return Maybe<void>::Ok();
  }

  Maybe<bool> NeedStaticScheduling(StageChainEdge* edge, int64_t num_stage_placement_ids) const {
    // stage_placement_id is different from parallel_desc_symbol_id. It's configured by user and
    // there is no related symbol about it.
    if (num_stage_placement_ids != 2) { return false; }
    const auto& src_pass_name = edge->src_node()->calculation_pass_name();
    if (!(src_pass_name == kForwardPass || src_pass_name == kBackwardPass)) { return false; }
    int64_t src_parallel_desc_symbol_id = edge->src_node()->parallel_desc_symbol_id();
    int64_t dst_parallel_desc_symbol_id = edge->dst_node()->parallel_desc_symbol_id();
    if (src_parallel_desc_symbol_id == dst_parallel_desc_symbol_id) { return true; }
    const auto& storage = *Global<vm::SymbolStorage<ParallelDesc>>::Get();
    const auto& src_parallel_desc = JUST(storage.MaybeGet(src_parallel_desc_symbol_id));
    const auto& dst_parallel_desc = JUST(storage.MaybeGet(dst_parallel_desc_symbol_id));
    const auto& src_device_tag = src_parallel_desc.parallel_conf().device_tag();
    const auto& dst_device_tag = dst_parallel_desc.parallel_conf().device_tag();
    if (src_device_tag == "cpu") { return false; }
    if (src_device_tag != dst_device_tag) { return false; }
    return src_parallel_desc.parallel_num() == dst_parallel_desc.parallel_num();
  }

  Maybe<void> ResetRegstNum(JobBuilder* job_builder, const std::string& op_name,
                            int64_t each_output_regst_num) const {
    auto* op_conf = JUST(job_builder->CachedMutOpConf4OpName(op_name));
    op_conf->set_each_output_regst_num(each_output_regst_num);
    return Maybe<void>::Ok();
  }

  Maybe<void> MakeGetterBufferSize4Lbi(
      const ComputeGraph& compute_graph,
      std::function<Maybe<int64_t>(const LogicalBlobId&)>* BufferSize4Lbi) const {
    auto lbi2buffer_size = std::make_shared<HashMap<LogicalBlobId, int64_t>>();
    JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& node) -> Maybe<void> {
      const auto& op = node.op();
      int64_t size = JUST(GetEachOutputRegstNum(op.op_conf()));
      for (const auto& obn : op.output_bns()) { (*lbi2buffer_size)[op.BnInOp2Lbi(obn)] = size; }
      return Maybe<void>::Ok();
    }));
    *BufferSize4Lbi = [lbi2buffer_size](const LogicalBlobId& lbi) -> Maybe<int64_t> {
      const auto& iter = lbi2buffer_size->find(lbi);
      CHECK_OR_RETURN(iter != lbi2buffer_size->end());
      return iter->second;
    };
    return Maybe<void>::Ok();
  }

  Maybe<int64_t> GetEachOutputRegstNum(const OperatorConf& op_conf) const {
    int64_t regst_num = 0;
    if (op_conf.has_each_output_regst_num()) { regst_num = op_conf.each_output_regst_num(); }
    if (op_conf.has_user_conf()) {
      const std::string& op_type_name = op_conf.user_conf().op_type_name();
      const auto* op_reg_result =
          user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
      CHECK_OR_RETURN(op_reg_result != nullptr)
          << "op_type_name " << op_type_name << " not register";
      if (op_reg_result->same_output_regst_num_getter) {
        user_op::UserOpConfWrapper user_op_conf(op_conf);
        regst_num = JUST((*op_reg_result->same_output_regst_num_getter)(user_op_conf));
      } else {
        regst_num = 1;
      }
    } else {
      regst_num = 1;
    }
    CHECK_GT_OR_RETURN(regst_num, 0);
    return regst_num;
  }

  Maybe<const Scope&> GetScope(const OperatorConf& op_conf) const {
    CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
    return Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(op_conf.scope_symbol_id());
  }

  Maybe<void> AddCtrlFromSrcSourceToDstSource(JobBuilder* job_builder,
                                              const ComputeGraph& compute_graph,
                                              const StageChainEdge* edge) const {
    JUST(edge->dst_node()->ForEachSourceComputeNode([&](const ComputeNode& dst) -> Maybe<void> {
      OperatorConf* op_conf = JUST(job_builder->CachedMutOpConf4OpName(dst.op().op_name()));
      JUST(edge->src_node()->ForEachSourceComputeNode([&](const ComputeNode& src) -> Maybe<void> {
        const auto& existed = op_conf->ctrl_in_op_name();
        const auto& op_name = src.op().op_name();
        if (std::find(existed.begin(), existed.end(), op_name) == existed.end()) {
          op_conf->add_ctrl_in_op_name(op_name);
        }
        return Maybe<void>::Ok();
      }));
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachLbi7RequiredBufferSize(
      const ComputeGraph& compute_graph, const StageChainGraph& stage_chain_graph,
      const std::function<Maybe<void>(const LogicalBlobId& lbi, int64_t required_buffer_size)>&
          DoEach) const {
    std::function<Maybe<bool>(const ComputeNode&)> IsDescendantOfAnyVar;
    JUST(MakePredicatorIsDescendantOfAnyVar(compute_graph, &IsDescendantOfAnyVar));
    const auto& scope_storage = *Global<vm::SymbolStorage<Scope>>::Get();
    std::function<Maybe<int64_t>(const OpBlobArg&)> RequiredBufferSize4Oba;
    JUST(MakeGetterRequiredBufferSize4Oba(stage_chain_graph, &RequiredBufferSize4Oba));
    JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& consumer_node) -> Maybe<void> {
      const auto& consumer_op = consumer_node.op();
      for (const auto& ibn : consumer_op.input_bns()) {
        const auto& input_modifier = consumer_op.InputBlobModifier4Ibn(ibn);
        if (input_modifier.is_mutable()) { continue; }
        if (input_modifier.use_header_only()) { continue; }
        const auto& lbi = consumer_op.BnInOp2Lbi(ibn);
        const auto& producer_node = JUST(compute_graph.Node4OpName(lbi.op_name()));
        if (producer_node.scope().scope_proto().calculation_pass_name() != kForwardPass) {
          continue;
        }
        if (consumer_node.scope().scope_proto().calculation_pass_name() != kForwardPass
            && !JUST(IsDescendantOfAnyVar(producer_node))) {
          continue;
        }
        const auto& producer_op = producer_node.op();
        if (producer_op.op_conf().has_variable_conf()) { continue; }
        int64_t required_buffer_size = 0;
        {
          OpBlobArg oba;
          oba.set_op_name(consumer_op.op_name());
          oba.set_bn_in_op(ibn);
          required_buffer_size = JUST(RequiredBufferSize4Oba(oba));
          int64_t src_scope_symbol_id = producer_op.op_conf().scope_symbol_id();
          const auto& src_scope = JUST(scope_storage.MaybeGet(src_scope_symbol_id));
          const auto& src_parallel_desc = JUST(src_scope.GetParallelDesc(producer_op.op_conf()));
          int64_t dst_scope_symbol_id = consumer_op.op_conf().scope_symbol_id();
          const auto& dst_scope = JUST(scope_storage.MaybeGet(dst_scope_symbol_id));
          const auto& dst_parallel_desc = JUST(dst_scope.GetParallelDesc(consumer_op.op_conf()));
          if (dst_parallel_desc != src_parallel_desc) {
            // copy op can be regarded as buffer op with buffer_size 1.
            --required_buffer_size;
          }
        }
        JUST(DoEach(lbi, required_buffer_size));
      }
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> MakePredicatorIsDescendantOfAnyVar(
      const ComputeGraph& compute_graph,
      std::function<Maybe<bool>(const ComputeNode&)>* IsDescendantOfAnyVar) const {
    auto var_desendants = std::make_shared<HashMap<const ComputeNode*, bool>>();
    *IsDescendantOfAnyVar = [var_desendants](const ComputeNode& node) -> Maybe<bool> {
      return MapAt(*var_desendants, &node);
    };
    compute_graph.TopoForEachNode([&](ComputeNode* node) {
      bool* is_desendant = &(*var_desendants)[node];
      node->ForEachNodeOnInEdge([&](ComputeNode* in_node) {
        *is_desendant = *is_desendant || (*var_desendants)[in_node];
        *is_desendant = *is_desendant || in_node->op().op_conf().has_variable_conf();
      });
    });
    return Maybe<void>::Ok();
  }

  Maybe<void> MakeGetterRequiredBufferSize4Oba(
      const StageChainGraph& stage_chain_graph,
      std::function<Maybe<int64_t>(const OpBlobArg&)>* RequiredBufferSize4Oba) const {
    auto oba2stage_chain_edge = std::make_shared<HashMap<OpBlobArg, StageChainEdge*>>();
    auto edge2required_buffer_size = std::make_shared<HashMap<StageChainEdge*, int64_t>>();
    *RequiredBufferSize4Oba = [oba2stage_chain_edge,
                               edge2required_buffer_size](const OpBlobArg& oba) -> Maybe<int64_t> {
      const auto& iter = oba2stage_chain_edge->find(oba);
      if (iter == oba2stage_chain_edge->end()) { return 0; }
      return MapAt(*edge2required_buffer_size, iter->second);
    };

    JUST(stage_chain_graph.MaybeForEachEdge([&](StageChainEdge* edge) -> Maybe<void> {
      // set oba2stage_chain_edge
      for (const auto* compute_node : edge->dst_node()->compute_nodes()) {
        for (const auto& ibn : compute_node->op().input_bns()) {
          const auto& lbi = compute_node->op().BnInOp2Lbi(ibn);
          if (edge->lbis().count(lbi) > 0) {
            OpBlobArg oba;
            oba.set_op_name(compute_node->op().op_name());
            oba.set_bn_in_op(ibn);
            CHECK_OR_RETURN(oba2stage_chain_edge->emplace(oba, edge).second);
          }
        }
      }
      // update edge2required_buffer_size
      int64_t required_buffer_size = JUST(edge->NumStagePlacementInPath());
      required_buffer_size = std::min(required_buffer_size, edge->src_node()->stage_buffer_size());
      (*edge2required_buffer_size)[edge] = required_buffer_size;
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("ResetStageRegstNum", ResetStageRegstNumPass);

}  // namespace

}  // namespace oneflow
