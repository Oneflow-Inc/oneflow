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
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/common/map_util.h"
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
    JobBuilder job_builder(job);
    return Apply(&job_builder);
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

  Maybe<void> Apply(JobBuilder* job_builder) const {
    HashMap<LogicalBlobId, StageBuffers> produced_lbi2stage_buffers;
    HashMap<std::string, std::shared_ptr<Operator>> op_name2op;
    JUST(job_builder->ForEachOperator([&](const std::shared_ptr<Operator>& op) -> Maybe<void> {
      CHECK_OR_RETURN(op_name2op.emplace(op->op_name(), op).second);
      return Maybe<void>::Ok();
    }));
    HashMap<const Operator*, StageBuffers> consumer_op2stage_buffers;
    JUST(ForEachStageBuffer(
        op_name2op, [&](const std::shared_ptr<StageBuffer>& stage_buffer) -> Maybe<void> {
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
      const HashMap<std::string, std::shared_ptr<Operator>>& op_name2op,
      const std::function<Maybe<void>(const std::shared_ptr<StageBuffer>&)>& DoEach) const {
    std::function<Maybe<int64_t>(const LogicalBlobId&)> BufferSize4Lbi;
    JUST(MakeGetterBufferSize4Lbi(op_name2op, &BufferSize4Lbi));
    JUST(ForEachLbi7Consumer7StagesInPath(
        op_name2op,
        [&](const LogicalBlobId& lbi, const Operator* consumer_op,
            const Range& path_stage_range) -> Maybe<void> {
          int64_t num_stages = path_stage_range.size();
          int64_t lbi_buffer_size = JUST(BufferSize4Lbi(lbi));
          CHECK_GT(lbi_buffer_size, 0);
          // buffer size provided: (stage_buffer->buffer_size + lbi_buffer_size)
          // buffer size required: num_stages
          // (stage_buffer->buffer_size + lbi_buffer_size) == num_stages
          if (num_stages <= lbi_buffer_size) { return Maybe<void>::Ok(); }
          auto stage_buffer = std::make_shared<StageBuffer>();
          stage_buffer->produced_lbi = lbi;
          {
            const auto& producer_op_conf = JUST(MapAt(op_name2op, lbi.op_name()))->op_conf();
            int64_t scope_symbol_id = producer_op_conf.scope_symbol_id();
            stage_buffer->scope_symbol_id = scope_symbol_id;
            stage_buffer->scope =
                &JUST(Global<vm::SymbolStorage<Scope>>::Get()->MaybeGet(scope_symbol_id));
          }
          stage_buffer->consumer_op = consumer_op;
          stage_buffer->buffer_size = num_stages - lbi_buffer_size;
          CHECK_GT_OR_RETURN(stage_buffer->buffer_size, 0);
          return DoEach(stage_buffer);
        }));
    return Maybe<void>::Ok();
  }

  Maybe<void> AddBufferOp(JobBuilder* job_builder, const LogicalBlobId& produced_lbi,
                          StageBuffers* stage_buffers) const {
    std::string op_name = produced_lbi.op_name() + "_buffer_op";
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
    job_builder->AddOps(parallel_desc.parallel_conf(), {buffer_op.op_conf()});
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
      const HashMap<std::string, std::shared_ptr<Operator>>& op_name2op,
      std::function<Maybe<int64_t>(const LogicalBlobId&)>* BufferSize4Lbi) const {
    auto lbi2buffer_size = std::make_shared<HashMap<LogicalBlobId, int64_t>>();
    for (const auto& pair : op_name2op) {
      const auto& op = pair.second;
      int64_t size = JUST(GetSameOutputRegstNum(op->op_conf()));
      for (const auto& obn : op->output_bns()) { (*lbi2buffer_size)[op->BnInOp2Lbi(obn)] = size; }
    }
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

  Maybe<bool> HasStageInfo(const OperatorConf& op_conf) const {
    const auto& scope = JUST(GetScope(op_conf));
    int64_t stage_id = scope.Int64("stage_id");
    int64_t num_stage = scope.Int64("num_stages");
    CHECK_LT_OR_RETURN(stage_id, num_stage);
    return stage_id >= 0 && num_stage > 0;
  }

  Maybe<int64_t> GetStageId(const OperatorConf& op_conf) const {
    const auto& scope = JUST(GetScope(op_conf));
    int64_t stage_id = scope.Int64("stage_id");
    CHECK_GE_OR_RETURN(stage_id, 0) << op_conf.DebugString();
    return stage_id;
  }

  Maybe<int64_t> GetNumStage(const OperatorConf& op_conf) const {
    const auto& scope = JUST(GetScope(op_conf));
    int64_t num_stage = scope.Int64("num_stages");
    CHECK_GE_OR_RETURN(num_stage, 0) << op_conf.DebugString();
    return num_stage;
  }

  // path_stage_range.end() is not in path_stage_range
  Maybe<void> ForEachLbi7Consumer7StagesInPath(
      const HashMap<std::string, std::shared_ptr<Operator>>& op_name2op,
      const std::function<Maybe<void>(const LogicalBlobId& lbi, const Operator* consumer_op,
                                      const Range& path_stage_range)>& DoEach) const {
    for (const auto& pair : op_name2op) {
      const auto& consumer_op = *pair.second;
      const auto& scope = JUST(GetScope(consumer_op.op_conf()));
      for (const auto& ibn : consumer_op.input_bns()) {
        const auto& input_modifier = consumer_op.InputBlobModifier4Ibn(ibn);
        if (input_modifier.is_mutable()) { continue; }
        if (input_modifier.use_header_only()) { continue; }
        const auto& lbi = consumer_op.BnInOp2Lbi(ibn);
        const auto& producer_op = *op_name2op.at(lbi.op_name());
        if (!JUST(HasStageInfo(producer_op.op_conf()))) { continue; }
        const auto& producer_scope = JUST(GetScope(producer_op.op_conf()));
        if (producer_scope.scope_proto().calculation_pass_name() == kForwardPass
            && scope.scope_proto().calculation_pass_name() != kForwardPass) {
          int64_t start_stage_id = JUST(GetStageId(producer_op.op_conf()));
          int64_t end_stage_id = JUST(GetNumStage(producer_op.op_conf()));
          CHECK_LT(start_stage_id, end_stage_id);
          Range path_stage_range(start_stage_id, end_stage_id);
          JUST(DoEach(lbi, &consumer_op, path_stage_range));
        } else {
          int64_t producer_stage_id = JUST(GetStageId(producer_op.op_conf()));
          int64_t consumer_stage_id = -1;
          if (JUST(HasStageInfo(consumer_op.op_conf()))) {
            consumer_stage_id = JUST(GetStageId(consumer_op.op_conf()));
          } else {
            consumer_stage_id = producer_stage_id + 1;
          }
          // path_stage_range.end() is not in range path_stage_range
          Range path_stage_range(producer_stage_id, consumer_stage_id);
          JUST(DoEach(lbi, &consumer_op, path_stage_range));
        }
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("AddStageBufferOp", AddStageBufferOpPass);

}  // namespace

}  // namespace oneflow
