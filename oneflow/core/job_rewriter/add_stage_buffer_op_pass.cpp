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
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"

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
    JobBuilder job_builder(job);
    return Apply(&job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_stage_buffer");
  }

  struct StageBuffer {
    LogicalBlobId produced_lbi;
    const Scope* scope;
    const Operator* consumer_op;
    size_t buffer_size;
    std::string buffer_op_out_lbn;
  };

  using StageBuffers = std::vector<std::shared_ptr<StageBuffer>>;

  Maybe<void> Apply(JobBuilder* job_builder) const {
    HashMap<LogicalBlobId, StageBuffers> produced_lbi2stage_buffers;
    HashMap<const Operator*, StageBuffers> consumer_op2stage_buffers;
    JUST(ForEachStageBuffer(*job_builder,
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

  Maybe<void> AddBufferOp(JobBuilder* job_builder, const LogicalBlobId& produced_lbi,
      StageBuffers* stage_buffers) const {
    std::string op_name = produced_lbi.op_name() + "_buffer_op";
    const Scope* scope = nullptr;
    size_t buffer_size = -1;
    for (const auto& stage_buffer : *stage_buffers) {
      if (scope == nullptr) {
        scope = stage_buffer.scope;
      } else {
        CHECK_EQ_OR_RETURN(scope, stage_buffer.scope);
      }
      if (buffer_size == -1) {
        buffer_size = stage_buffer.buffer_size;
      } else {
        CHECK_EQ_OR_RETURN(buffer_size, stage_buffer.buffer_size);
      }
    }
    const auto buffer_op = user_op::UserOpConfWrapperBuilder(op_name)
                              .Op("buffer")
                              .ScopeSymbolId(scope->scope_proto().symbol_id())
                              .Input("in", GenLogicalBlobName(produced_lbi))
                              .Output("out")
                              .Attr<int64_t>("buffer_size", buffer_size)
                              .Build();
    const auto& parallel_desc = JUST(scope->GetParallelDesc(buffer_op.op_conf()));
    job_builder->AddOps(parallel_desc.parallel_conf(), {buffer_op.op_conf()});
    for (auto& stage_buffer : *stage_buffers) {
      stage_buffer.buffer_op_out_lbn = op_name + "/out_0";
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> ReplaceInputWithBufferOutLbn(
      JobBuilder* job_builder, const Operator& op, const StageBuffers& stage_buffers) const {
    const auto& FindStageBuffer = [&](const LogicalBlobId& lbi) -> const StageBuffer* {
      const auto& iter = std::find_if(stage_buffers.begin(), stage_buffers.end(),
          [&](const std::shared_ptr<StageBuffer>& stage_buffer) {
            return stage_buffer->produced_lbi == lbi;
          });
      if (iter == stage_buffers.end()) { return nullptr; }
      return iter->get();
    };
    std::unique_ptr<std::vector<Operator>> new_op_confs;
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      const StageBuffer* stage_buffer = FindStageBuffer(lbi);
      if (stage_buffer == nullptr) { continue; }
      if (!new_op_confs) { new_op_confs.reset(new std::vector<OperatorConf>({op.op_conf()})); }
      auto* new_op_conf = &new_op_confs->at(0);
      ReplaceInputLbnInOpCustomizedConf(new_op_conf, ibn, stage_buffer->buffer_op_out_lbn);
    }
    if (new_op_confs) { job_builder->MutOpsOnlyOnce(*new_op_confs); }
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachTrainableVarOpNode(const OpGraph& op_graph,
                                        const std::function<Maybe<void>(OpNode*)>& DoEach) const {
    std::function<bool(OpNode*)> NeedBackwardOp;
    JUST(MakePredicatorNeedBackwardOp(op_graph, &NeedBackwardOp));
    const auto& IsSspVarProxy = [](const Operator& op_conf) {
      return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == "ssp_variable_proxy";
    };
    JUST(op_graph.MaybeForEachNode([&](OpNode* op_node) -> Maybe<void> {
      const auto& op_conf = op_node->op().op_conf();
      CHECK_OR_RETURN(!IsSspVarProxy(op_conf)) << "AddStageBufferOp can not be applied twice";
      if (op_conf.has_variable_conf() && NeedBackwardOp(op_node)) { return DoEach(op_node); }
      return Maybe<void>::Ok();
    }));
    return Maybe<void>::Ok();
  }

  Maybe<void> ReplaceVarWithSspVarProxyOp(
      OpNode* op_node, JobBuilder* job_builder,
      const std::function<bool(const LogicalBlobId&)>& NeedReplace,
      const std::function<const std::string&(const LogicalBlobId&)>& Ref4Var,
      const std::function<const std::string&(const LogicalBlobId&)>& Val4Var) const {
    const auto& op = op_node->op();
    std::unique_ptr<std::vector<Operator>> new_op_confs;
    for (const auto& ibn : op.input_bns()) {
      const auto& lbi = op.BnInOp2Lbi(ibn);
      if (!NeedReplace(lbi)) { continue; }
      if (!new_op_confs) { new_op_confs.reset(new std::vector<OperatorConf>({op.op_conf()})); }
      auto* new_op_conf = &new_op_confs->at(0);
      int64_t scope_symbol_id = op.op_conf().scope_symbol_id();
      bool in_optimizer_pass = JUST(IsInOptimizerPass(scope_symbol_id));
      const auto* lbn = (in_optimizer_pass ? &Ref4Var(lbi) : &Val4Var(lbi));
      ReplaceInputLbnInOpCustomizedConf(new_op_conf, ibn, *lbn);
    }
    if (new_op_confs) { job_builder->MutOpsOnlyOnce(*new_op_confs); }
    return Maybe<void>::Ok();
  }

};

REGISTER_JOB_PASS("AddStageBufferOp", AddStageBufferOpPass);

}  // namespace

}  // namespace oneflow
