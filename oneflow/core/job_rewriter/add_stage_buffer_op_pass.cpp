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

class AddStageBufferOpPass final : public JobPass {
 public:
  AddStageBufferOpPass(const AddStageBufferOpPass&) = delete;
  AddStageBufferOpPass(AddStageBufferOpPass&&) = delete;
  AddStageBufferOpPass() = default;
  ~AddStageBufferOpPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    auto compute_graph = JUST(ComputeGraph::New(*job));
    auto stage_chain_graph = JUST(StageChainGraph::New(*compute_graph));
    {
      std::string job_name = ctx->job_desc().job_name();
      compute_graph->ToDotWithFilePath(std::string("compute_graph-") + job_name + ".dot");
      stage_chain_graph->ToDotWithFilePath(std::string("stage_chain_graph-") + job_name + ".dot");
    }
    JobBuilder job_builder(job);
    return Apply(*compute_graph, *stage_chain_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_stage_buffer");
  }

  struct StageBuffer {
    LogicalBlobId produced_lbi;
    int64_t scope_symbol_id;
    const Scope* scope;
    const Operator* consumer_op;
    int64_t buffer_size;

    // set after buffer_op added
    std::string buffer_op_out_lbn;
  };

  using StageBuffers = std::vector<std::shared_ptr<StageBuffer>>;

  Maybe<void> Apply(const ComputeGraph& compute_graph, const StageChainGraph& stage_chain_graph,
                    JobBuilder* job_builder) const {
    HashMap<LogicalBlobId, StageBuffers> produced_lbi2stage_buffers;
    HashMap<const Operator*, StageBuffers> consumer_op2stage_buffers;
    JUST(ForEachStageBuffer(
        compute_graph, stage_chain_graph,
        [&](const std::shared_ptr<StageBuffer>& stage_buffer) -> Maybe<void> {
          CHECK_GT_OR_RETURN(stage_buffer->buffer_size, 0);
          produced_lbi2stage_buffers[stage_buffer->produced_lbi].push_back(stage_buffer);
          consumer_op2stage_buffers[stage_buffer->consumer_op].push_back(stage_buffer);
          return Maybe<void>::Ok();
        }));
    for (auto& pair : produced_lbi2stage_buffers) {
      JUST(AddBufferOp(job_builder, pair.first, &pair.second));
    }
    for (const auto& pair : consumer_op2stage_buffers) {
      JUST(ReplaceInputWithBufferOutLbn(job_builder, *pair.first, pair.second));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachStageBuffer(
      const ComputeGraph& compute_graph, const StageChainGraph& stage_chain_graph,
      const std::function<Maybe<void>(const std::shared_ptr<StageBuffer>&)>& DoEach) const {
    std::function<Maybe<int64_t>(const LogicalBlobId&)> BufferSize4Lbi;
    JUST(MakeGetterBufferSize4Lbi(compute_graph, &BufferSize4Lbi));
    JUST(ForEachLbi7Consumer7RequiredBufferSize(
        compute_graph, stage_chain_graph,
        [&](const LogicalBlobId& lbi, const Operator* consumer_op,
            int64_t required_buffer_size) -> Maybe<void> {
          int64_t lbi_buffer_size = JUST(BufferSize4Lbi(lbi));
          CHECK_GT(lbi_buffer_size, 0);
          // buffer size provided: (stage_buffer->buffer_size + lbi_buffer_size)
          // buffer size required: required_buffer_size
          // (stage_buffer->buffer_size + lbi_buffer_size) == required_buffer_size
          if (required_buffer_size <= lbi_buffer_size) { return Maybe<void>::Ok(); }
          auto stage_buffer = std::make_shared<StageBuffer>();
          stage_buffer->produced_lbi = lbi;
          {
            const auto& producer_op = JUST(compute_graph.Node4OpName(lbi.op_name())).op();
            int64_t scope_symbol_id = producer_op.op_conf().scope_symbol_id();
            stage_buffer->scope_symbol_id = scope_symbol_id;
            stage_buffer->scope =
                &JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(scope_symbol_id));
          }
          stage_buffer->consumer_op = consumer_op;
          stage_buffer->buffer_size = required_buffer_size - lbi_buffer_size;
          CHECK_GT_OR_RETURN(stage_buffer->buffer_size, 0);
          return DoEach(stage_buffer);
        }));
    return Maybe<void>::Ok();
  }

  Maybe<void> AddBufferOp(JobBuilder* job_builder, const LogicalBlobId& produced_lbi,
                          StageBuffers* stage_buffers) const {
    std::string op_name = produced_lbi.op_name() + "__" + produced_lbi.blob_name() + "__buffer_op";
    const Scope* scope = nullptr;
    int64_t scope_symbol_id = 0;
    int64_t buffer_size = -1;
    CHECK_OR_RETURN(!stage_buffers->empty());
    for (const auto& stage_buffer : *stage_buffers) {
      if (scope == nullptr) {
        scope = stage_buffer->scope;
        scope_symbol_id = stage_buffer->scope_symbol_id;
      } else {
        CHECK_EQ_OR_RETURN(scope, stage_buffer->scope);
      }
      buffer_size = std::max(buffer_size, stage_buffer->buffer_size);
    }
    CHECK_GT_OR_RETURN(buffer_size, 0);
    const auto buffer_op = user_op::UserOpConfWrapperBuilder(op_name)
                               .Op("buffer")
                               .ScopeSymbolId(scope_symbol_id)
                               .Input("in", GenLogicalBlobName(produced_lbi))
                               .Output("out")
                               .Attr<int64_t>("buffer_size", buffer_size)
                               .Build();
    const auto& parallel_desc = JUST(scope->GetParallelDesc(buffer_op.op_conf()));
    OF_RETURN_IF_ERROR(job_builder->AddOps(parallel_desc.parallel_conf(), {buffer_op.op_conf()}))
        << buffer_op.op_conf().DebugString();
    for (auto& stage_buffer : *stage_buffers) {
      stage_buffer->buffer_op_out_lbn = op_name + "/out_0";
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> ReplaceInputWithBufferOutLbn(JobBuilder* job_builder, const Operator& op,
                                           const StageBuffers& stage_buffers) const {
    const auto& FindStageBuffer = [&](const LogicalBlobId& lbi) -> const StageBuffer* {
      const auto& iter = std::find_if(stage_buffers.begin(), stage_buffers.end(),
                                      [&](const std::shared_ptr<StageBuffer>& stage_buffer) {
                                        return stage_buffer->produced_lbi == lbi;
                                      });
      if (iter == stage_buffers.end()) { return nullptr; }
      return iter->get();
    };
    std::unique_ptr<OperatorConf> new_op_conf;
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      const StageBuffer* stage_buffer = FindStageBuffer(lbi);
      if (stage_buffer == nullptr) { continue; }
      if (op.InputBlobModifier4Ibn(ibn).is_mutable()) { continue; }
      if (op.InputBlobModifier4Ibn(ibn).use_header_only()) { continue; }
      if (!new_op_conf) { new_op_conf.reset(new OperatorConf(op.op_conf())); }
      ReplaceInputLbnInOpCustomizedConf(new_op_conf.get(), ibn, stage_buffer->buffer_op_out_lbn);
    }
    if (new_op_conf) { job_builder->MutOpsOnlyOnce({*new_op_conf}); }
    return Maybe<void>::Ok();
  }

  Maybe<void> MakeGetterBufferSize4Lbi(
      const ComputeGraph& compute_graph,
      std::function<Maybe<int64_t>(const LogicalBlobId&)>* BufferSize4Lbi) const {
    auto lbi2buffer_size = std::make_shared<HashMap<LogicalBlobId, int64_t>>();
    JUST(compute_graph.ForEachComputeNode([&](const ComputeNode& node) -> Maybe<void> {
      const auto& op = node.op();
      int64_t size = JUST(GetSameOutputRegstNum(op.op_conf()));
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

  Maybe<int64_t> GetSameOutputRegstNum(const OperatorConf& op_conf) const {
    if (op_conf.has_user_conf()) {
      const std::string& op_type_name = op_conf.user_conf().op_type_name();
      const auto* op_reg_result =
          user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
      CHECK_OR_RETURN(op_reg_result != nullptr)
          << "op_type_name " << op_type_name << " not register";
      if (op_reg_result->same_output_regst_num_getter) {
        user_op::UserOpConfWrapper user_op_conf(op_conf);
        return JUST((*op_reg_result->same_output_regst_num_getter)(user_op_conf));
      } else {
        return 1;
      }
    } else {
      return 1;
    }
  }

  Maybe<const Scope&> GetScope(const OperatorConf& op_conf) const {
    CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
    return Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(op_conf.scope_symbol_id());
  }

  Maybe<void> ForEachLbi7Consumer7RequiredBufferSize(
      const ComputeGraph& compute_graph, const StageChainGraph& stage_chain_graph,
      const std::function<Maybe<void>(const LogicalBlobId& lbi, const Operator* consumer_op,
                                      int64_t required_buffer_size)>& DoEach) const {
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
        JUST(DoEach(lbi, &consumer_op, required_buffer_size));
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

    HashMap<StageChainEdge*, HashSet<int64_t>> edge2path_parallel_desc_symbol_ids;
    auto IsReachable = stage_chain_graph.MakePredicatorIsReachable();
    stage_chain_graph.ForEachEdge([&](StageChainEdge* edge) {
      auto* src = edge->src_node();
      auto* dst = edge->dst_node();
      // set oba2stage_chain_edge
      for (const auto* compute_node : dst->compute_nodes()) {
        for (const auto& ibn : compute_node->op().input_bns()) {
          const auto& lbi = compute_node->op().BnInOp2Lbi(ibn);
          if (edge->lbis().count(lbi) > 0) {
            OpBlobArg oba;
            oba.set_op_name(compute_node->op().op_name());
            oba.set_bn_in_op(ibn);
            CHECK(oba2stage_chain_edge->emplace(oba, edge).second);
          }
        }
      }
      // update edge2path_parallel_desc_symbol_ids
      auto* path_parallel_desc_symbol_ids = &edge2path_parallel_desc_symbol_ids[edge];
      const auto& ForEachNext = [&](StageChainNode* node,
                                    const std::function<void(StageChainNode*)>& DoEach) {
        node->ForEachNodeOnInOutEdge([&](StageChainNode* next) {
          if (IsReachable(src, next) && IsReachable(next, dst)) { DoEach(next); }
        });
      };
      stage_chain_graph.BfsForEachNode({src}, ForEachNext, [&](StageChainNode* node) {
        path_parallel_desc_symbol_ids->insert(node->parallel_desc_symbol_ids().begin(),
                                              node->parallel_desc_symbol_ids().end());
      });
    });
    for (const auto& pair : edge2path_parallel_desc_symbol_ids) {
      auto* edge = pair.first;
      int64_t required_buffer_size = pair.second.size();
      required_buffer_size = std::min(required_buffer_size, edge->src_node()->stage_buffer_size());
      (*edge2required_buffer_size)[edge] = required_buffer_size;
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("AddStageBufferOp", AddStageBufferOpPass);

}  // namespace

}  // namespace oneflow
