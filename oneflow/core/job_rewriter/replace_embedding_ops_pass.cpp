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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/dynamic_loss_scale_job_pass_state.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/embedding/embedding_options.h"

namespace oneflow {

namespace {

constexpr char const* kOptimizerConfStateKey = "ONE_EMBEDDING_OPTIMIZE_CONF";
constexpr char const* kOptimizerConfPlaceholderPrefix = "one_embedding_optimizer_placeholder::";

struct OneEmbeddingOptimizerState : public JobPassState {
  HashMap<std::string, OptimizerConf> name2conf;
};

class DumpOneEmbeddingOptimizerConfPass final : public JobPass {
 public:
  DumpOneEmbeddingOptimizerConfPass() = default;
  ~DumpOneEmbeddingOptimizerConfPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> DumpOneEmbeddingOptimizerConfPass::Apply(Job* job, JobPassCtx* ctx) const {
  std::unique_ptr<OneEmbeddingOptimizerState> state(new OneEmbeddingOptimizerState);
  for (auto& opt_conf : *job->mutable_job_conf()->mutable_train_conf()->mutable_optimizer_conf()) {
    PbRpf<std::string> new_variable_names;
    for (const auto& name : opt_conf.variable_op_names()) {
      const size_t pos = name.find(kOptimizerConfPlaceholderPrefix);
      if (pos == std::string::npos) {
        *new_variable_names.Add() = name;
        continue;
      }
      const std::string embedding_name = name.substr(pos + strlen(kOptimizerConfPlaceholderPrefix));
      state->name2conf.emplace(embedding_name, opt_conf);
      LOG(ERROR) << "Find Embedding " << embedding_name;
    }
    *opt_conf.mutable_variable_op_names() = new_variable_names;
  }
  JUST(ctx->ResetState(kOptimizerConfStateKey, std::move(state)));
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("DumpOneEmbeddingOptimizerConfPass", DumpOneEmbeddingOptimizerConfPass);

std::string BuildIdentityOp(JobBuilder* job_builder, const std::string& in_lbn,
                            const ParallelConf& parallel_conf,
                            const user_op::UserOpConfWrapper& embedding_op) {
  user_op::UserOpConfWrapperBuilder identity_op_builder(embedding_op.op_name() + "_identity_"
                                                        + NewUniqueId());
  user_op::UserOpConfWrapper identity_op =
      identity_op_builder.OpTypeName("identity")
          .Input("in", in_lbn)
          .Output("out")
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  job_builder->AddOps(parallel_conf, {identity_op.op_conf()});
  return identity_op.output("out", 0);
}

void DynamicLossScaleAddGradient(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                                 const std::vector<std::string>& gradient_lbns,
                                 int64_t scope_symbol_id, const ParallelConf& parallel_conf) {
  if (job_builder->job().job_conf().train_conf().has_dynamic_loss_scale_policy()) {
    CHECK_GT(gradient_lbns.size(), 0);
    const auto& dynamic_loss_scale_state =
        CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    const LogicalBlobId count_not_finite_lbi =
        GenLogicalBlobId(dynamic_loss_scale_state.count_not_finite_lbn());
    const OpNode* op_node = op_graph.OpNode4OpName(count_not_finite_lbi.op_name());  // identity
    if (op_node->op().op_conf().has_user_conf()
        && op_node->op().op_conf().user_conf().op_type_name() == "identity") {
      const user_op::UserOpConfWrapper identity_op_conf(op_node->op().op_conf());
      std::string new_count_not_finite_lbn;
      if (gradient_lbns.size() == 1) {
        const auto count_not_finite_op =
            user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-CountNotFinite-"
                                              + NewUniqueId())
                .Op("count_not_finite")
                .Input("x", gradient_lbns.at(0))
                .Output("y")
                .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id())
                .Build();
        job_builder->AddOps(parallel_conf, {count_not_finite_op.op_conf()});
        new_count_not_finite_lbn = count_not_finite_op.output("y", 0);
      } else {
        auto multi_count_not_finite_op_builder =
            user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-MultiCountNotFinite-"
                                              + NewUniqueId())
                .Op("multi_count_not_finite")
                .Output("y")
                .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id());
        for (const auto& lbn : gradient_lbns) { multi_count_not_finite_op_builder.Input("x", lbn); }
        const auto multi_count_not_finite_op = multi_count_not_finite_op_builder.Build();
        job_builder->AddOps(parallel_conf, {multi_count_not_finite_op.op_conf()});
        new_count_not_finite_lbn = multi_count_not_finite_op.output("y", 0);
      }
      user_op::UserOpConfWrapperBuilder add_op_builder("System-DynamicLossScale-CountNotFinite-Add_"
                                                       + NewUniqueId());
      const auto add_op = add_op_builder.Op("add_n")
                              .Input("in", identity_op_conf.input("in", 0))
                              .Input("in", new_count_not_finite_lbn)
                              .Output("out")
                              .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id())
                              .Build();

      job_builder->AddOps(op_node->parallel_desc().parallel_conf(), {add_op.op_conf()});

      OperatorConf new_identity_conf = identity_op_conf.op_conf();
      const auto& old_val =
          ReplaceInputLbnInOpCustomizedConf(&new_identity_conf, "in_0", add_op.output("out", 0));
      CHECK_EQ(identity_op_conf.input("in", 0), old_val);
      job_builder->MutOpsOnlyOnce({new_identity_conf});

    } else {
      UNIMPLEMENTED();
    }
  }
}

std::string AddScheduleOp(const OpGraph& op_graph, JobBuilder* job_builder,
                          const OptimizerConf& optimizer_conf, const std::string& op_name) {
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  const class oneflow::OpNode* op_node =
      op_graph.OpNode4OpName(GenLogicalBlobId(train_conf.train_step_lbn()).op_name());
  CHECK_OR_RETURN(op_node != nullptr) << "op node not found in op graph, op name: " << op_name;
  const ParallelConf& parallel_conf = op_node->parallel_desc().parallel_conf();
  OperatorConf schedule_op_conf{};
  schedule_op_conf.set_name(op_name);
  auto* schedule_conf = schedule_op_conf.mutable_learning_rate_schedule_conf();
  schedule_conf->set_train_step(train_conf.train_step_lbn());
  schedule_conf->set_learning_rate(optimizer_conf.base_learning_rate());
  schedule_conf->set_out("out");
  if (optimizer_conf.has_warmup_conf()) {
    *schedule_conf->mutable_warmup_conf() = optimizer_conf.warmup_conf();
  }
  if (optimizer_conf.has_learning_rate_decay()) {
    *schedule_conf->mutable_learning_rate_decay() = optimizer_conf.learning_rate_decay();
  }
  schedule_op_conf.set_scope_symbol_id(op_node->op().op_conf().scope_symbol_id());
  job_builder->AddOps(parallel_conf, {schedule_op_conf});
  return GenLogicalBlobName(op_name, schedule_conf->out());
}

void BuildEmbeddingLookup(JobPassCtx* ctx, JobBuilder* job_builder, const int64_t embedding_size,
                          const int64_t line_size, const std::string& embedding_name,
                          const ParallelConf& parallel_conf,
                          const user_op::UserOpConfWrapper& embedding_op,
                          const user_op::UserOpConfWrapper& id_shuffle_op,
                          std::string* embedding_lbn, std::string* unique_values_lbn) {
  auto AddIdentityOp = [&](const std::string& in_lbn) -> std::string {
    return BuildIdentityOp(job_builder, in_lbn, parallel_conf, embedding_op);
  };
  const std::string unique_ids_lbn = id_shuffle_op.output("cur_rank_unique_ids", 0);
  // embedding prefetch op
  user_op::UserOpConfWrapperBuilder embedding_prefetch_op_builder(embedding_op.op_name()
                                                                  + "_embedding_prefetch");
  user_op::UserOpConfWrapper embedding_prefetch_op =
      embedding_prefetch_op_builder.OpTypeName("embedding_prefetch")
          .Input("num_unique_ids", id_shuffle_op.output("cur_rank_num_unique_ids", 0))
          .Input("unique_ids", unique_ids_lbn)
          .Input("column_ids", id_shuffle_op.output("cur_rank_column_ids", 0))
          .Output("context")
          .Attr<int64_t>("embedding_size", embedding_size)
          .Attr<int64_t>("line_size", line_size)
          .Attr<std::string>("embedding_options",
                             embedding_op.attr<std::string>("embedding_options"))
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  OperatorConf embedding_prefetch_new_op_conf = embedding_prefetch_op.op_conf();
  embedding_prefetch_new_op_conf.set_stream_name_hint("EMBEDDING");
  job_builder->AddOps(parallel_conf, {embedding_prefetch_new_op_conf});

  // embedding lookup op
  user_op::UserOpConfWrapperBuilder embedding_lookup_op_builder(embedding_op.op_name()
                                                                + "_embedding_lookup");
  embedding_lookup_op_builder.OpTypeName("embedding_lookup")
      .Input("num_unique_ids", AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
      .Input("unique_ids", AddIdentityOp(unique_ids_lbn))
      .Input("context", embedding_prefetch_op.output("context", 0))
      .Output("unique_values")
      .Attr<DataType>("dtype", embedding_op.attr<DataType>("dtype"))
      .Attr<int64_t>("embedding_size", embedding_size)
      .Attr<int64_t>("line_size", line_size)
      .Attr<std::string>("embedding_name", embedding_name)
      .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id());
  bool has_embeddings_output =
      (line_size != embedding_size) || ctx->job_desc().enable_auto_mixed_precision();
  if (has_embeddings_output) {
    DataType embeddings_dtype = ctx->job_desc().enable_auto_mixed_precision()
                                    ? DataType::kFloat16
                                    : embedding_op.attr<DataType>("dtype");
    embedding_lookup_op_builder.Output("embeddings")
        .Attr<DataType>("embeddings_dtype", embeddings_dtype);
  }
  user_op::UserOpConfWrapper embedding_lookup_op = embedding_lookup_op_builder.Build();
  OperatorConf embedding_lookup_new_op_conf = embedding_lookup_op.op_conf();
  embedding_lookup_new_op_conf.set_stream_name_hint("EMBEDDING");
  job_builder->AddOps(parallel_conf, {embedding_lookup_new_op_conf});
  if (has_embeddings_output) {
    *embedding_lbn = embedding_lookup_op.output("embeddings", 0);
  } else {
    *embedding_lbn = embedding_lookup_op.output("unique_values", 0);
  }
  *unique_values_lbn = embedding_lookup_op.output("unique_values", 0);

  // cast
  // if (ctx->job_desc().enable_auto_mixed_precision()) {
  //  auto cast_op = user_op::UserOpConfWrapperBuilder(embedding_op.op_name() + "_cast_f2h")
  //                     .Op("cast")
  //                     .Input("in", *embedding_lbn)
  //                     .Output("out")
  //                     .Attr<DataType>("dtype", DataType::kFloat16)
  //                     .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
  //                     .Build();
  //  *embedding_lbn = cast_op.output("out", 0);
  //  job_builder->AddOps(parallel_conf, {cast_op.op_conf()});
  //}
}

void BuildEmbeddingShuffle(JobBuilder* job_builder, const ParallelConf& parallel_conf,
                           const user_op::UserOpConfWrapper& embedding_op,
                           const user_op::UserOpConfWrapper& id_shuffle_op,
                           const std::string& embedding_lbn) {
  auto AddIdentityOp = [&](const std::string& in_lbn) -> std::string {
    return BuildIdentityOp(job_builder, in_lbn, parallel_conf, embedding_op);
  };
  // embedding shuffle op
  user_op::UserOpConfWrapperBuilder embedding_shuffle_op_builder(embedding_op.op_name());
  user_op::UserOpConfWrapper embedding_shuffle_op =
      embedding_shuffle_op_builder.OpTypeName("embedding_shuffle")
          .Input("cur_rank_embeddings", embedding_lbn)
          .Input("cur_rank_reverse_idx",
                 AddIdentityOp(id_shuffle_op.output("cur_rank_reverse_idx", 0)))
          .Input("ids_reverse_idx", AddIdentityOp(id_shuffle_op.output("ids_reverse_idx", 0)))
          .Input("num_unique_ids_matrix",
                 AddIdentityOp(id_shuffle_op.output("num_unique_ids_matrix", 0)))
          .Input("partition_index", AddIdentityOp(id_shuffle_op.output("partition_index", 0)))
          .Output("embeddings")
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  // add_ops.push_back(embedding_shuffle_op.op_conf());
  // delete_op_names.push_back(embedding_op.op_name());
  job_builder->MutOpOnlyOnce(embedding_shuffle_op.op_conf());
}

void BuildEmbeddingGradientShuffle(JobPassCtx* ctx, const OpGraph& op_graph,
                                   JobBuilder* job_builder, const ParallelConf& parallel_conf,
                                   const user_op::UserOpConfWrapper& embedding_op,
                                   const user_op::UserOpConfWrapper& id_shuffle_op,
                                   const std::string& update_embedding_diff,
                                   std::string* cur_rank_unique_embedding_diff_lbn) {
  auto AddIdentityOp = [&](const std::string& in_lbn) -> std::string {
    return BuildIdentityOp(job_builder, in_lbn, parallel_conf, embedding_op);
  };
  std::string update_embedding_diff_lbn = update_embedding_diff;
  if (ctx->job_desc().enable_auto_mixed_precision()
      && ParseBooleanFromEnv("GRADIENT_SHUFFLE_USE_FP16", true)) {
    LogicalBlobId embedding_diff_lbi = GenLogicalBlobId(update_embedding_diff_lbn);
    const OpNode* cast_node = op_graph.OpNode4OpName(embedding_diff_lbi.op_name());
    if (cast_node->op().op_conf().has_user_conf()) {
      const user_op::UserOpConfWrapper cast_op_conf(cast_node->op().op_conf());
      if (cast_op_conf.op_type_name() == "cast") {
        update_embedding_diff_lbn = cast_op_conf.input("in", 0);
        job_builder->DelOps({cast_op_conf.op_name()});
      }
    }
  }
  // embedding_gradient_shuffle op
  user_op::UserOpConfWrapperBuilder embedding_gradient_shuffle_op_builder(
      embedding_op.op_name() + "_embedding_gradient_shuffle");
  user_op::UserOpConfWrapper embedding_gradient_shuffle_op =
      embedding_gradient_shuffle_op_builder.OpTypeName("embedding_gradient_shuffle")
          .Input("cur_rank_reverse_idx",
                 AddIdentityOp(id_shuffle_op.output("cur_rank_reverse_idx", 0)))
          .Input("ids_reverse_idx", AddIdentityOp(id_shuffle_op.output("ids_reverse_idx", 0)))
          .Input("embedding_diff", update_embedding_diff_lbn)
          .Input("num_unique_ids_matrix",
                 AddIdentityOp(id_shuffle_op.output("num_unique_ids_matrix", 0)))
          .Input("partition_index", AddIdentityOp(id_shuffle_op.output("partition_index", 0)))
          .Output("cur_rank_unique_embedding_diff")
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  job_builder->AddOps(parallel_conf, {embedding_gradient_shuffle_op.op_conf()});
  *cur_rank_unique_embedding_diff_lbn =
      embedding_gradient_shuffle_op.output("cur_rank_unique_embedding_diff", 0);

  if (ctx->job_desc().enable_auto_mixed_precision()
      && ParseBooleanFromEnv("GRADIENT_SHUFFLE_USE_FP16", true)
      && ParseBooleanFromEnv("NOT_FUSE_CAST_TO_UPDATE", false)) {
    auto cast_op = user_op::UserOpConfWrapperBuilder(embedding_op.op_name() + "_cast_h2f")
                       .Op("cast")
                       .Input("in", *cur_rank_unique_embedding_diff_lbn)
                       .Output("out")
                       .Attr<DataType>("dtype", DataType::kFloat)
                       .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
                       .Build();
    *cur_rank_unique_embedding_diff_lbn = cast_op.output("out", 0);
    job_builder->AddOps(parallel_conf, {cast_op.op_conf()});
  }
}

double GetLossInstanceNumScaleFactor(const OpGraph& op_graph, JobBuilder* job_builder) {
  double scale_factor = 1;
  std::function<OpNode*(const std::string&)> LossOpNode4OpName;
  CHECK_JUST(MakeGetterLossOpNode4OpName(op_graph, &LossOpNode4OpName));
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  HashMap<LogicalBlobId, OpNode*> loss_lbi2op_node;
  for (const auto& loss_lbn : train_conf.loss_lbn()) {
    const auto& lbi = GenLogicalBlobId(loss_lbn);
    CHECK(loss_lbi2op_node.emplace(lbi, LossOpNode4OpName(lbi.op_name())).second);
  }
  const Shape src_time_shape({1, 1});
  const int64_t source_time_shape_elem_cnt = src_time_shape.elem_cnt();
  bool all_loss_time_shape_eq_src = true;
  for (const auto& pair : loss_lbi2op_node) {
    const int64_t time_shape_elem_cnt = CHECK_JUST(pair.second->op().GetOpTimeShape())->elem_cnt();
    if (time_shape_elem_cnt != source_time_shape_elem_cnt) {
      CHECK_EQ(time_shape_elem_cnt % source_time_shape_elem_cnt, 0);
      all_loss_time_shape_eq_src = false;
    }
  }
  if (all_loss_time_shape_eq_src) {
    const BlobDesc* blob_desc = nullptr;
    for (const auto& pair : loss_lbi2op_node) {
      const BlobDesc* cur_blob_desc = &pair.second->LogicalBlobDesc4Lbi(pair.first);
      if (blob_desc != nullptr) { CHECK(*blob_desc == *cur_blob_desc); }
      blob_desc = cur_blob_desc;
    }
    CHECK(!blob_desc->is_dynamic());
    scale_factor = 1.0f / static_cast<float>(blob_desc->shape().elem_cnt());
  } else {
    std::unique_ptr<BlobDesc> blob_desc;
    for (const auto& pair : loss_lbi2op_node) {
      const BlobDesc* cur_blob_desc = &pair.second->LogicalBlobDesc4Lbi(pair.first);
      // TODO: support dynamic
      CHECK(!cur_blob_desc->is_dynamic());
      const DataType loss_data_type = cur_blob_desc->data_type();
      const int64_t time_shape_elem_cnt =
          CHECK_JUST(pair.second->op().GetOpTimeShape())->elem_cnt();
      // TODO: consider sbp
      const int64_t loss_elem_cnt =
          cur_blob_desc->shape().elem_cnt() * time_shape_elem_cnt / source_time_shape_elem_cnt;
      if (blob_desc) {
        CHECK_EQ(blob_desc->data_type(), loss_data_type);
        CHECK_EQ(blob_desc->shape().elem_cnt(), loss_elem_cnt);
      } else {
        blob_desc.reset(new BlobDesc(Shape({loss_elem_cnt}), loss_data_type));
      }
    }
    scale_factor = 1.0f / static_cast<float>(blob_desc->shape().elem_cnt());
  }
  return scale_factor;
}

void BuildEmbeddingUpdate(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                          const ParallelConf& parallel_conf, const int64_t embedding_size,
                          const std::string& embedding_name, const OptimizerConf& optimizer_conf,
                          const user_op::UserOpConfWrapper& embedding_op,
                          const user_op::UserOpConfWrapper& id_shuffle_op,
                          const std::string& unique_values_lbn,
                          const std::string& embedding_diff_lbn,
                          const std::string& learning_rate_lbn) {
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  auto AddIdentityOp = [&](const std::string& in_lbn) -> std::string {
    return BuildIdentityOp(job_builder, in_lbn, parallel_conf, embedding_op);
  };
  auto AddAdamBiasCorrectionFactorOp = [&](float beta_val,
                                           const std::string& op_name) -> std::string {
    user_op::UserOpConfWrapperBuilder op_builder(embedding_op.op_name() + op_name);
    const auto adam_bias_correction_factor_op =
        op_builder.OpTypeName("adam_bias_correction_factor")
            .Input("train_step", train_conf.train_step_lbn())
            .Attr<float>("beta", beta_val)
            .Output("out")
            .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
            .Build();
    job_builder->AddOps(parallel_conf, {adam_bias_correction_factor_op.op_conf()});
    return adam_bias_correction_factor_op.output("out", 0);
  };
  user_op::UserOpConfWrapperBuilder embedding_update_op_builder(embedding_op.op_name()
                                                                + "_embedding_update");
  if (optimizer_conf.has_naive_conf()) {
    embedding_update_op_builder.OpTypeName("sgd_embedding_update");
  } else if (optimizer_conf.has_momentum_conf()) {
    embedding_update_op_builder.OpTypeName("momentum_embedding_update")
        .Attr<float>("beta", optimizer_conf.momentum_conf().beta());
  } else if (optimizer_conf.has_adam_conf()) {
    const AdamModelUpdateConf& adam_conf = optimizer_conf.adam_conf();
    embedding_update_op_builder.OpTypeName("adam_embedding_update")
        .Attr<float>("beta1", adam_conf.beta1())
        .Attr<float>("beta2", adam_conf.beta2())
        .Attr<float>("epsilon", adam_conf.epsilon())
        .Attr<bool>("do_bias_correction", adam_conf.do_bias_correction());
    if (adam_conf.do_bias_correction()) {
      const std::string bias_correction1_lbn =
          AddAdamBiasCorrectionFactorOp(adam_conf.beta1(), "adam_bias_correction_factor1");
      const std::string bias_correction2_lbn =
          AddAdamBiasCorrectionFactorOp(adam_conf.beta2(), "adam_bias_correction_factor2");
      embedding_update_op_builder.Input("bias_correction1", bias_correction1_lbn)
          .Input("bias_correction2", bias_correction2_lbn);
    }
  } else {
    UNIMPLEMENTED();
  }
  embedding_update_op_builder
      .Input("num_unique_ids", AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
      .Input("unique_embeddings", unique_values_lbn)
      .Input("embedding_diff", embedding_diff_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Output("updated_unique_embeddings");
  double scale = GetLossInstanceNumScaleFactor(op_graph, job_builder);
  if (train_conf.has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    embedding_update_op_builder.Input("skip_if", dynamic_loss_scale_state.count_not_finite_lbn());
    embedding_update_op_builder.Input("scale_by_tensor",
                                      dynamic_loss_scale_state.loss_scale_val_lbn());
  } else if (train_conf.has_loss_scale_factor()) {
    double down_scale_factor = 1.0f / train_conf.loss_scale_factor();
    scale *= down_scale_factor;
  }
  embedding_update_op_builder.Attr<double>("scale", scale);
  user_op::UserOpConfWrapper embedding_update_op =
      embedding_update_op_builder.Attr<int64_t>("embedding_size", embedding_size)
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  OperatorConf embedding_update_new_op_conf = embedding_update_op.op_conf();
  embedding_update_new_op_conf.set_stream_name_hint("EMBEDDING");
  job_builder->AddOps(parallel_conf, {embedding_update_new_op_conf});

  user_op::UserOpConfWrapperBuilder embedding_put_op_builder(embedding_op.op_name()
                                                             + "_embedding_put");
  user_op::UserOpConfWrapper embedding_put_op =
      embedding_put_op_builder.OpTypeName("embedding_put")
          .Input("num_unique_ids",
                 AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
          .Input("unique_ids", AddIdentityOp(id_shuffle_op.output("cur_rank_unique_ids", 0)))
          .Input("unique_embeddings", embedding_update_op.output("updated_unique_embeddings", 0))
          .Attr<std::string>("embedding_name", embedding_name)
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  OperatorConf embedding_put_new_op_conf = embedding_put_op.op_conf();
  embedding_put_new_op_conf.set_stream_name_hint("EMBEDDING");
  job_builder->AddOps(parallel_conf, {embedding_put_new_op_conf});
}

}  // namespace

class ReplaceEmbeddingOps final : public JobPass {
 public:
  ReplaceEmbeddingOps() = default;
  ~ReplaceEmbeddingOps() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return true; }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder, JobPassCtx* ctx) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder, ctx);
  }
};

Maybe<void> ReplaceEmbeddingOps::Apply(const OpGraph& op_graph, JobBuilder* job_builder,
                                       JobPassCtx* ctx) const {
  std::vector<std::string> gradient_lbns;
  ParallelConf embedding_parallel_conf;
  int64_t embedding_scope_symbol_id;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (!(op_conf.user_conf().op_type_name() == "embedding_lookup_placeholder")) { return; }
    const user_op::UserOpConfWrapper embedding_op(op_node->op().op_conf());
    const LogicalBlobId& lbi = GenLogicalBlobId(embedding_op.input("shadow", 0));
    const std::string& shadow_op_name = lbi.op_name();
    embedding::EmbeddingOptions options(embedding_op.attr<std::string>("embedding_options"));
    std::vector<OperatorConf> add_ops;
    std::vector<std::string> delete_op_names;

    // id shuffle op
    user_op::UserOpConfWrapperBuilder id_shuffle_op_builder(embedding_op.op_name() + "_id_shuffle");
    user_op::UserOpConfWrapper id_shuffle_op =
        id_shuffle_op_builder.OpTypeName("id_shuffle")
            .Input("ids", embedding_op.input("ids", 0))
            .Input("column_ids", embedding_op.input("column_ids", 0))
            .Output("num_unique_ids")
            .Output("ids_reverse_idx")
            .Output("cur_rank_num_unique_ids")
            .Output("cur_rank_unique_ids")
            .Output("cur_rank_column_ids")
            .Output("cur_rank_reverse_idx")
            .Output("num_unique_ids_matrix")
            .Output("partition_index")
            .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
            .Build();
    OperatorConf id_shuffle_new_op_conf = id_shuffle_op.op_conf();
    id_shuffle_new_op_conf.set_stream_name_hint("ID_SHUFFLE");
    add_ops.push_back(id_shuffle_new_op_conf);

    // embedding lookup op
    std::string embedding_lbn, unique_values_lbn;
    BuildEmbeddingLookup(ctx, job_builder, options.EmbeddingSize(), options.LineSize(),
                         options.Name(), op_node->parallel_desc().parallel_conf(), embedding_op,
                         id_shuffle_op, &embedding_lbn, &unique_values_lbn);

    // embedding shuffle op
    BuildEmbeddingShuffle(job_builder, op_node->parallel_desc().parallel_conf(), embedding_op,
                          id_shuffle_op, embedding_lbn);

    // find update op
    const OpNode* producer =
        op_graph.OpNode4OpName(GenLogicalBlobId(embedding_op.input("ids", 0)).op_name());
    for (OpEdge* edge : producer->out_edges()) {
      const OpNode* consumer = edge->dst_node();
      if (consumer->op().op_conf().has_user_conf()) {
        const user_op::UserOpConfWrapper update_op_conf(consumer->op().op_conf());
        if (update_op_conf.op_type_name() != "embedding_update_placeholder") { continue; }
        if (update_op_conf.attr<std::string>("embedding_options")
            != embedding_op.attr<std::string>("embedding_options")) {
          continue;
        }
        delete_op_names.push_back(update_op_conf.op_name());

        std::string embedding_diff_lbn;
        BuildEmbeddingGradientShuffle(
            ctx, op_graph, job_builder, op_node->parallel_desc().parallel_conf(), embedding_op,
            id_shuffle_op, update_op_conf.input("embedding_diff", 0), &embedding_diff_lbn);

        // dynamic loss scale
        gradient_lbns.push_back(embedding_diff_lbn);
        // assert all embeddings same placement
        embedding_scope_symbol_id = embedding_op.op_conf().scope_symbol_id();
        embedding_parallel_conf = op_node->parallel_desc().parallel_conf();

        OptimizerConf embedding_optimizer_conf;
        bool found_embedding_optimizer = false;
        for (const auto& optimizer_conf :
             job_builder->job().job_conf().train_conf().optimizer_conf()) {
          for (const auto& name : optimizer_conf.variable_op_names()) {
            if (name == shadow_op_name) {
              embedding_optimizer_conf = optimizer_conf;
              found_embedding_optimizer = true;
              break;
            }
          }
          if (found_embedding_optimizer == true) { break; }
        }
        CHECK_EQ(found_embedding_optimizer, true);
        const std::string& learning_rate_lbn =
            AddScheduleOp(op_graph, job_builder, embedding_optimizer_conf,
                          "System-Train-LearningRate-Scheduler_" + NewUniqueId());

        BuildEmbeddingUpdate(ctx, op_graph, job_builder, op_node->parallel_desc().parallel_conf(),
                             options.EmbeddingSize(), options.Name(), embedding_optimizer_conf,
                             embedding_op, id_shuffle_op, unique_values_lbn, embedding_diff_lbn,
                             learning_rate_lbn);
      }
    }
    job_builder->DelOps(delete_op_names);
    job_builder->AddOps(op_node->parallel_desc().parallel_conf(), add_ops);
  });
  if (gradient_lbns.size() > 0) {
    DynamicLossScaleAddGradient(ctx, op_graph, job_builder, gradient_lbns,
                                embedding_scope_symbol_id, embedding_parallel_conf);
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceEmbeddingOps", ReplaceEmbeddingOps);

}  // namespace oneflow
