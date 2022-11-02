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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/job_rewriter/clip_by_global_norm_job_pass_state.h"
#include "oneflow/core/embedding/embedding_manager.h"

namespace oneflow {

namespace {

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

Maybe<void> DynamicLossScaleAddGradient(
    JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
    const HashMap<std::string, std::string>& shadow_op_name2grad_lbn, int64_t scope_symbol_id,
    const ParallelConf& parallel_conf) {
  if (job_builder->job().job_conf().train_conf().has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    const LogicalBlobId count_not_finite_lbi =
        GenLogicalBlobId(dynamic_loss_scale_state.count_not_finite_lbn());
    const OpNode* op_node = op_graph.OpNode4OpName(count_not_finite_lbi.op_name());
    if (op_node->op().op_conf().has_user_conf()
        && op_node->op().op_conf().user_conf().op_type_name() == "identity") {
      const user_op::UserOpConfWrapper identity_op_conf(op_node->op().op_conf());
      std::string new_count_not_finite_lbn;
      if (shadow_op_name2grad_lbn.size() == 1) {
        const std::string& grad_lbn = shadow_op_name2grad_lbn.begin()->second;
        const auto count_not_finite_op =
            user_op::UserOpConfWrapperBuilder("OneEmbedding-DynamicLossScale-CountNotFinite-"
                                              + NewUniqueId())
                .Op("count_not_finite")
                .Input("x", grad_lbn)
                .Output("y")
                .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id())
                .Build();
        job_builder->AddOps(parallel_conf, {count_not_finite_op.op_conf()});
        new_count_not_finite_lbn = count_not_finite_op.output("y", 0);
      } else {
        auto multi_count_not_finite_op_builder =
            user_op::UserOpConfWrapperBuilder("OneEmbedding-DynamicLossScale-MultiCountNotFinite-"
                                              + NewUniqueId())
                .Op("multi_count_not_finite")
                .Output("y")
                .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id());
        for (const auto& pair : shadow_op_name2grad_lbn) {
          multi_count_not_finite_op_builder.Input("x", pair.second);
        }
        const auto multi_count_not_finite_op = multi_count_not_finite_op_builder.Build();
        job_builder->AddOps(parallel_conf, {multi_count_not_finite_op.op_conf()});
        new_count_not_finite_lbn = multi_count_not_finite_op.output("y", 0);
      }
      user_op::UserOpConfWrapperBuilder add_op_builder(
          "OneEmbedding-DynamicLossScale-CountNotFinite-Add_" + NewUniqueId());
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
      CHECK_EQ_OR_RETURN(identity_op_conf.input("in", 0), old_val);
      job_builder->MutOpsOnlyOnce({new_identity_conf});
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  return Maybe<void>::Ok();
}

void BuildEmbeddingLookup(
    JobPassCtx* ctx, JobBuilder* job_builder, const int64_t embedding_size, const int64_t line_size,
    const std::string& embedding_name, const int64_t seed, bool has_embedding_prefetch,
    const ParallelConf& parallel_conf, const user_op::UserOpConfWrapper& embedding_op,
    const std::string& prefetch_num_unique_ids_lbn, const std::string& prefetch_unique_ids_lbn,
    const std::string& prefetch_unique_table_ids_lbn, const std::string& num_unique_ids_lbn,
    const std::string& unique_ids_lbn, const std::string& unique_table_ids_lbn,
    std::string* embedding_lbn, std::string* unique_values_lbn,
    OperatorConf* embedding_prefetch_op_conf, OperatorConf* embedding_lookup_op_conf) {
  std::string context_lbn;
  if (has_embedding_prefetch) {
    // embedding prefetch op
    user_op::UserOpConfWrapperBuilder embedding_prefetch_op_builder(
        embedding_op.op_name() + "_embedding_prefetch" + NewUniqueId());
    user_op::UserOpConfWrapper embedding_prefetch_op =
        embedding_prefetch_op_builder.OpTypeName("embedding_prefetch")
            .Input("num_unique_ids", prefetch_num_unique_ids_lbn)
            .Input("unique_ids", prefetch_unique_ids_lbn)
            .Input("table_ids", prefetch_unique_table_ids_lbn)
            .Output("context")
            .Attr<int64_t>("embedding_size", embedding_size)
            .Attr<int64_t>("line_size", line_size)
            .Attr<std::string>("embedding_tables",
                               embedding_op.attr<std::string>("embedding_tables"))
            .Attr<std::string>("embedding_name", embedding_name)
            .Attr<int64_t>("seed", seed)
            .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
            .Build();
    *embedding_prefetch_op_conf = embedding_prefetch_op.op_conf();
    if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
      embedding_prefetch_op_conf->set_stream_name_hint(embedding_name + "_EMBEDDING");
    }
    context_lbn = embedding_prefetch_op.output("context", 0);
  }

  // embedding lookup op
  user_op::UserOpConfWrapperBuilder embedding_lookup_op_builder(
      embedding_op.op_name() + "_embedding_lookup" + NewUniqueId());
  embedding_lookup_op_builder.OpTypeName("embedding_lookup")
      .Input("num_unique_ids", num_unique_ids_lbn)
      .Input("unique_ids", unique_ids_lbn)
      .Input("table_ids", unique_table_ids_lbn)
      .Output("unique_values")
      .Attr<DataType>("dtype", embedding_op.attr<DataType>("dtype"))
      .Attr<int64_t>("embedding_size", embedding_size)
      .Attr<int64_t>("line_size", line_size)
      .Attr<std::string>("embedding_tables", embedding_op.attr<std::string>("embedding_tables"))
      .Attr<std::string>("embedding_name", embedding_name)
      .Attr<int64_t>("seed", seed)
      .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id());
  if (has_embedding_prefetch) { embedding_lookup_op_builder.Input("context", context_lbn); }
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
  *embedding_lookup_op_conf = embedding_lookup_op.op_conf();
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
    embedding_lookup_op_conf->set_stream_name_hint(embedding_name + "_EMBEDDING");
  }
  if (has_embeddings_output) {
    *embedding_lbn = embedding_lookup_op.output("embeddings", 0);
  } else {
    *embedding_lbn = embedding_lookup_op.output("unique_values", 0);
  }
  *unique_values_lbn = embedding_lookup_op.output("unique_values", 0);
}

void BuildEmbeddingShuffle(JobBuilder* job_builder, const std::string& embedding_name,
                           int64_t embedding_size, const ParallelConf& parallel_conf,
                           const user_op::UserOpConfWrapper& embedding_op,
                           const std::string& inverse_indices_lbn,
                           const std::string& inner_inverse_unique_partition_indices_lbn,
                           const std::string& num_unique_matrix_lbn,
                           const std::string& embedding_lbn, std::vector<OperatorConf>* add_ops,
                           std::string* new_embeddings_lbn) {
  const bool is_train_job = job_builder->job().job_conf().has_train_conf();
  user_op::UserOpConfWrapperBuilder embedding_shuffle_op_builder(
      embedding_op.op_name() + "_embedding_shuffle" + NewUniqueId());
  user_op::UserOpConfWrapper embedding_shuffle_op =
      embedding_shuffle_op_builder.OpTypeName("embedding_shuffle")
          .Input("cur_rank_embeddings", embedding_lbn)
          .Input("cur_rank_inverse_indices", inverse_indices_lbn)
          .Input("inverse_unique_partition_indices", inner_inverse_unique_partition_indices_lbn)
          .Input("num_unique_matrix", num_unique_matrix_lbn)
          .Attr<std::string>("embedding_name", embedding_name)
          .Attr<int64_t>("embedding_size", embedding_size)
          .Attr<bool>("is_train", is_train_job)
          .Output("embeddings")
          .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
          .Build();
  OperatorConf embedding_shuffle_new_op_conf = embedding_shuffle_op.op_conf();
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)
      && ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_EMBEDDING_SHUFFLE_INDEPENTENT_STREAM", true)) {
    embedding_shuffle_new_op_conf.set_stream_name_hint(embedding_name + "_EMBEDDING");
  }
  add_ops->push_back(embedding_shuffle_new_op_conf);
  *new_embeddings_lbn = embedding_shuffle_op.output("embeddings", 0);
}

void BuildEmbeddingGradientShuffle(
    JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder, const OpNode* op_node,
    const std::string& embedding_name, int64_t embedding_size, const bool use_system_gather,
    const ParallelConf& embedding_parallel_conf, const int64_t embedding_scope_symbol_id,
    const user_op::UserOpConfWrapper& embedding_op, const std::string& inverse_indices_lbn,
    const std::string& inner_inverse_unique_partition_indices_lbn,
    const std::string& num_unique_matrix_lbn, const std::string& update_embedding_grad,
    const bool has_clip_grad, std::string* cur_rank_unique_embedding_grad_lbn) {
  std::string update_embedding_grad_lbn = update_embedding_grad;
  if (ctx->job_desc().enable_auto_mixed_precision()
      && !ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_GRADIENT_SHUFFLE_USE_FP16", true)) {
    auto cast_op =
        user_op::UserOpConfWrapperBuilder(embedding_op.op_name() + "_before_grad_shuffle_cast_h2f")
            .Op("cast")
            .Input("in", update_embedding_grad_lbn)
            .Output("out")
            .Attr<DataType>("dtype", DataType::kFloat)
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    job_builder->AddOps(embedding_parallel_conf, {cast_op.op_conf()});
    update_embedding_grad_lbn = cast_op.output("out", 0);
  }
  if (use_system_gather) {
    const int64_t num_segments =
        op_node->LogicalBlobDesc4Lbi(op_node->op().BnInOp2Lbi("ids_0")).shape().elem_cnt();
    user_op::UserOpConfWrapperBuilder unsorted_segment_sum_op_builder(embedding_op.op_name()
                                                                      + "_unsorted_segment_sum");
    user_op::UserOpConfWrapper unsorted_segment_sum_op =
        unsorted_segment_sum_op_builder.OpTypeName("unsorted_segment_sum")
            .Input("data", update_embedding_grad_lbn)
            .Input("segment_ids", inverse_indices_lbn)
            .Output("out")
            .Attr<int64_t>("num_segments", num_segments)
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    job_builder->AddOps(embedding_parallel_conf, {unsorted_segment_sum_op.op_conf()});
    *cur_rank_unique_embedding_grad_lbn = unsorted_segment_sum_op.output("out", 0);
  } else {
    // embedding_gradient_shuffle op
    // if no dynamic loss scale or no clip_grad, we think gradient shuffle grad's invalid buffer
    // need not to be memset.
    const bool has_dynamic_loss_scale =
        job_builder->job().job_conf().train_conf().has_dynamic_loss_scale_policy();
    const bool only_zero_valid_grad = (!has_clip_grad) && (!has_dynamic_loss_scale);
    user_op::UserOpConfWrapperBuilder embedding_gradient_shuffle_op_builder(
        embedding_op.op_name() + "_embedding_gradient_shuffle" + NewUniqueId());
    user_op::UserOpConfWrapper embedding_gradient_shuffle_op =
        embedding_gradient_shuffle_op_builder.OpTypeName("embedding_gradient_shuffle")
            .Input("cur_rank_inverse_indices", inverse_indices_lbn)
            .Input("inverse_unique_partition_indices", inner_inverse_unique_partition_indices_lbn)
            .Input("embedding_grad", update_embedding_grad_lbn)
            .Input("num_unique_matrix", num_unique_matrix_lbn)
            .Output("cur_rank_unique_embedding_grad")
            .Attr<std::string>("embedding_name", embedding_name)
            .Attr<int64_t>("embedding_size", embedding_size)
            .Attr<bool>("only_zero_valid_grad", only_zero_valid_grad)
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    OperatorConf embedding_gradient_shuffle_new_op_conf = embedding_gradient_shuffle_op.op_conf();
    if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)
        && ParseBooleanFromEnv(
            "ONEFLOW_ONE_EMBEDDING_EMBEDDING_GRADIENT_SHUFFLE_INDEPENTENT_STREAM", true)) {
      embedding_gradient_shuffle_new_op_conf.set_stream_name_hint(embedding_name + "_EMBEDDING");
    }
    job_builder->AddOps(embedding_parallel_conf, {embedding_gradient_shuffle_new_op_conf});
    *cur_rank_unique_embedding_grad_lbn =
        embedding_gradient_shuffle_op.output("cur_rank_unique_embedding_grad", 0);
  }
  if (ctx->job_desc().enable_auto_mixed_precision()
      && ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_GRADIENT_SHUFFLE_USE_FP16", true)
      && (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_NOT_FUSE_CAST_TO_UPDATE", false)
          || has_clip_grad)) {
    auto cast_op = user_op::UserOpConfWrapperBuilder(embedding_op.op_name() + "_cast_h2f")
                       .Op("cast")
                       .Input("in", *cur_rank_unique_embedding_grad_lbn)
                       .Output("out")
                       .Attr<DataType>("dtype", DataType::kFloat)
                       .ScopeSymbolId(embedding_scope_symbol_id)
                       .Build();
    *cur_rank_unique_embedding_grad_lbn = cast_op.output("out", 0);
    job_builder->AddOps(embedding_parallel_conf, {cast_op.op_conf()});
  }
}

double GetLossInstanceNumScaleFactor(const OpGraph& op_graph, JobBuilder* job_builder) {
  double scale_factor = 1;
  std::function<OpNode*(const std::string&)> LossOpNode4OpName;
  CHECK_JUST(MakeGetterLossOpNode4OpName(op_graph, &LossOpNode4OpName));
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  HashMap<LogicalBlobId, OpNode*> loss_lbi2op_node;
  CHECK_GT(train_conf.loss_lbn().size(), 0);
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
    CHECK(blob_desc != nullptr);
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

void BuildIdShuffle(bool use_system_gather, const std::string& embedding_name,
                    const user_op::UserOpConfWrapper& embedding_op,
                    std::vector<OperatorConf>* add_ops, std::string* prefetch_num_unique_lbn,
                    std::string* prefetch_unique_ids_lbn,
                    std::string* prefetch_unique_table_ids_lbn,
                    std::string* inner_inverse_unique_partition_indices_lbn,
                    std::string* num_unique_ids_lbn, std::string* unique_ids_lbn,
                    std::string* unique_table_ids_lbn, std::string* inverse_indices_lbn,
                    std::string* num_unique_matrix_lbn) {
  const int32_t num_tables = embedding_op.attr<int32_t>("num_tables");
  const int64_t padding_idx = embedding_op.attr<int64_t>("padding_idx");
  const int64_t has_padding_idx = embedding_op.attr<bool>("has_padding_idx");
  bool enable_pipelined_execution =
      !ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false);
  if (use_system_gather) {
    user_op::UserOpConfWrapperBuilder unique_op_builder(embedding_op.op_name()
                                                        + "_unique_ids_and_tables");
    unique_op_builder.OpTypeName("unique_key_value_pair")
        .Input("keys", embedding_op.input("ids", 0))
        .Output("num_unique")
        .Output("unique_keys")
        .Output("unique_values")
        .Output("inverse_indices")
        .Attr<int32_t>("num_tables", num_tables)
        .Attr<int64_t>("padding_idx", padding_idx)
        .Attr<bool>("has_padding_idx", has_padding_idx)
        .Attr<std::string>("embedding_name", embedding_name)
        .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id());
    if (embedding_op.has_input("table_ids", 0)) {
      unique_op_builder.Input("values", embedding_op.input("table_ids", 0));
    }
    user_op::UserOpConfWrapper unique_op = unique_op_builder.Build();
    OperatorConf unique_new_op_conf = unique_op.op_conf();
    if (enable_pipelined_execution) {
      unique_new_op_conf.set_stream_name_hint(embedding_name + "_ID_SHUFFLE");
    }
    add_ops->push_back(unique_new_op_conf);
    *num_unique_ids_lbn = unique_op.output("num_unique", 0);
    *unique_ids_lbn = unique_op.output("unique_keys", 0);
    *unique_table_ids_lbn = unique_op.output("unique_values", 0);
    *inverse_indices_lbn = unique_op.output("inverse_indices", 0);
    *prefetch_num_unique_lbn = *num_unique_ids_lbn;
    *prefetch_unique_ids_lbn = *unique_ids_lbn;
    *prefetch_unique_table_ids_lbn = *unique_table_ids_lbn;
  } else {
    user_op::UserOpConfWrapperBuilder id_shuffle_op_builder(embedding_op.op_name() + "_id_shuffle"
                                                            + NewUniqueId());
    id_shuffle_op_builder.OpTypeName("id_shuffle")
        .Input("ids", embedding_op.input("ids", 0))
        .Output("inverse_unique_partition_indices")
        .Output("cur_rank_num_unique")
        .Output("cur_rank_unique_ids")
        .Output("cur_rank_unique_table_ids")
        .Output("cur_rank_inverse_indices")
        .Output("num_unique_matrix")
        .Attr<int32_t>("num_tables", num_tables)
        .Attr<int64_t>("padding_idx", padding_idx)
        .Attr<bool>("has_padding_idx", has_padding_idx)
        .Attr<std::string>("embedding_name", embedding_name)
        .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id());
    if (embedding_op.has_input("table_ids", 0)) {
      id_shuffle_op_builder.Input("table_ids", embedding_op.input("table_ids", 0));
    }
    user_op::UserOpConfWrapper id_shuffle_op = id_shuffle_op_builder.Build();
    OperatorConf id_shuffle_new_op_conf = id_shuffle_op.op_conf();
    if (enable_pipelined_execution) {
      id_shuffle_new_op_conf.set_stream_name_hint(embedding_name + "_ID_SHUFFLE");
    }
    add_ops->push_back(id_shuffle_new_op_conf);
    if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ADD_ID_SHUFFLE_COPY_OUT", true)) {
      // add id_shuffle_copy_out, so the consumer can use light_actor and cuda_graph.
      user_op::UserOpConfWrapperBuilder identity_op_builder(
          embedding_op.op_name() + "_id_shuffle_copy_out_" + NewUniqueId());
      user_op::UserOpConfWrapper identity_op =
          identity_op_builder.OpTypeName("id_shuffle_copy_out")
              .Attr<std::string>("embedding_name", embedding_name)
              .Input("inverse_unique_partition_indices",
                     id_shuffle_op.output("inverse_unique_partition_indices", 0))
              .Input("cur_rank_num_unique", id_shuffle_op.output("cur_rank_num_unique", 0))
              .Input("cur_rank_unique_ids", id_shuffle_op.output("cur_rank_unique_ids", 0))
              .Input("cur_rank_unique_table_ids",
                     id_shuffle_op.output("cur_rank_unique_table_ids", 0))
              .Input("cur_rank_inverse_indices",
                     id_shuffle_op.output("cur_rank_inverse_indices", 0))
              .Input("num_unique_matrix", id_shuffle_op.output("num_unique_matrix", 0))
              .Output("out_inverse_unique_partition_indices")
              .Output("out_cur_rank_num_unique")
              .Output("out_cur_rank_unique_ids")
              .Output("out_cur_rank_unique_table_ids")
              .Output("out_cur_rank_inverse_indices")
              .Output("out_num_unique_matrix")
              .ScopeSymbolId(embedding_op.op_conf().scope_symbol_id())
              .Build();
      OperatorConf identity_op_conf = identity_op.op_conf();
      if (enable_pipelined_execution) {
        identity_op_conf.set_stream_name_hint(embedding_name + "_EMBEDDING");
      }
      add_ops->push_back(identity_op_conf);
      *inner_inverse_unique_partition_indices_lbn =
          identity_op.output("out_inverse_unique_partition_indices", 0);
      *num_unique_ids_lbn = identity_op.output("out_cur_rank_num_unique", 0);
      *unique_ids_lbn = identity_op.output("out_cur_rank_unique_ids", 0);
      *unique_table_ids_lbn = identity_op.output("out_cur_rank_unique_table_ids", 0);
      *inverse_indices_lbn = identity_op.output("out_cur_rank_inverse_indices", 0);
      *num_unique_matrix_lbn = identity_op.output("out_num_unique_matrix", 0);
    } else {
      *inner_inverse_unique_partition_indices_lbn =
          id_shuffle_op.output("inverse_unique_partition_indices", 0);
      *num_unique_ids_lbn = id_shuffle_op.output("cur_rank_num_unique", 0);
      *unique_ids_lbn = id_shuffle_op.output("cur_rank_unique_ids", 0);
      *unique_table_ids_lbn = id_shuffle_op.output("cur_rank_unique_table_ids", 0);
      *inverse_indices_lbn = id_shuffle_op.output("cur_rank_inverse_indices", 0);
      *num_unique_matrix_lbn = id_shuffle_op.output("num_unique_matrix", 0);
    }
    *prefetch_num_unique_lbn = id_shuffle_op.output("cur_rank_num_unique", 0);
    *prefetch_unique_ids_lbn = id_shuffle_op.output("cur_rank_unique_ids", 0);
    *prefetch_unique_table_ids_lbn = id_shuffle_op.output("cur_rank_unique_table_ids", 0);
  }
}

void MakeConstantInitializerAttr(const int64_t embedding_size, const int64_t line_size,
                                 const std::vector<float>& values, std::string* initializer_attr) {
  if (embedding_size == line_size) { return; }
  const int32_t num_states = line_size / embedding_size - 1;
  CHECK_GT(num_states, 0) << "num_states " << num_states;
  CHECK(values.size() == 0 || num_states == values.size())
      << "must set " << num_states << " optimizer states init value, but get " << values.size();
  nlohmann::json initializers;
  for (int32_t i = 0; i < num_states; ++i) {
    nlohmann::json initializer;
    initializer["type"] = "constant";
    const float initial_value = values.size() > 0 ? values.at(i) : 0.0;
    initializer["value"] = initial_value;
    initializers.push_back(initializer);
  }
  *initializer_attr = initializers.dump();
}

void ScaleGrad(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
               const ParallelConf& embedding_parallel_conf, const int64_t embedding_scope_symbol_id,
               const bool has_clip_grad, const std::string& embedding_grad_lbn,
               std::string* new_embedding_grad_lbn, std::string* update_skip_if_lbn,
               std::string* fuse_to_update_down_scale_by_lbn, double* fuse_to_update_scale) {
  *new_embedding_grad_lbn = embedding_grad_lbn;
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  double scale = GetLossInstanceNumScaleFactor(op_graph, job_builder);
  if (train_conf.has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    const std::string& loss_scale_val_lbn = dynamic_loss_scale_state.loss_scale_val_lbn();
    *update_skip_if_lbn = dynamic_loss_scale_state.count_not_finite_lbn();
    if (has_clip_grad) {
      const LogicalBlobId loss_scale_val_lbi = GenLogicalBlobId(loss_scale_val_lbn);
      const OpNode* loss_scale_node = op_graph.OpNode4OpName(loss_scale_val_lbi.op_name());
      auto inv_scale_op = user_op::UserOpConfWrapperBuilder(
                              "OneEmbedding-DynamicLossScale-Reciprocal-" + NewUniqueId())
                              .Op("reciprocal")
                              .Input("x", loss_scale_val_lbn)
                              .Output("y")
                              .ScopeSymbolId(loss_scale_node->op().op_conf().scope_symbol_id())
                              .Build();
      job_builder->AddOps(loss_scale_node->parallel_desc().parallel_conf(),
                          {inv_scale_op.op_conf()});

      auto scalar_mul_op = user_op::UserOpConfWrapperBuilder(
                               "OneEmbedding-ModelDiffScale-ScalarMul-" + NewUniqueId())
                               .Op("scalar_mul_by_tensor")
                               .Input("x", *new_embedding_grad_lbn)
                               .Input("scalar", inv_scale_op.output("y", 0))
                               .Output("y")
                               .ScopeSymbolId(embedding_scope_symbol_id)
                               .Build();
      job_builder->AddOps(embedding_parallel_conf, {scalar_mul_op.op_conf()});
      *new_embedding_grad_lbn = scalar_mul_op.output("y", 0);
    } else {
      *fuse_to_update_down_scale_by_lbn = loss_scale_val_lbn;
    }
  } else if (train_conf.has_loss_scale_factor()) {
    double down_scale_factor = 1.0f / train_conf.loss_scale_factor();
    scale *= down_scale_factor;
  }
  if (has_clip_grad) {
    auto scalar_mul_op =
        user_op::UserOpConfWrapperBuilder("OneEmbedding-ModelDiffScale-ScalarMul-" + NewUniqueId())
            .Op("scalar_mul")
            .Input("in", *new_embedding_grad_lbn)
            .Output("out")
            .Attr<bool>("has_float_operand", true)
            .Attr<double>("float_operand", scale)
            .Attr<bool>("has_int_operand", false)
            .Attr<int64_t>("int_operand", 0)
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    job_builder->AddOps(embedding_parallel_conf, {scalar_mul_op.op_conf()});
    *new_embedding_grad_lbn = scalar_mul_op.output("out", 0);
    *fuse_to_update_scale = 1.0;
  } else {
    *fuse_to_update_scale = scale;
  }
}

bool IsSupportFusedUpdatePut(const bool is_full_cache, const bool enable_auto_mixed_precision,
                             const bool is_sgd, const std::string& down_scale_by_lbn,
                             const std::string& skip_if_lbn, const float l1, const float l2,
                             const float weight_decay) {
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSE_UPDATE_PUT", true)) { return false; }
  if (!is_full_cache) { return false; }
  if (!enable_auto_mixed_precision) { return false; }
  if (!is_sgd) { return false; }
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_GRADIENT_SHUFFLE_USE_FP16", true)) {
    return false;
  }
  if (!down_scale_by_lbn.empty()) { return false; }
  if (!skip_if_lbn.empty()) { return false; }
  if (l1 != 0) { return false; }
  if (l2 != 0) { return false; }
  if (weight_decay != 0) { return false; }
  return true;
}

void BuildEmbeddingUpdate(
    JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
    const ParallelConf& embedding_parallel_conf, const int64_t embedding_scope_symbol_id,
    const bool is_full_cache, const int64_t embedding_size, const int64_t line_size, const float l1,
    const float l2, const std::string& embedding_name, const OptimizerConf& optimizer_conf,
    const user_op::UserOpConfWrapper& embedding_op, const std::string& num_unique_ids_lbn,
    const std::string& unique_ids_lbn, const std::string& unique_values_lbn,
    const std::string& embedding_grad_lbn, const std::string& learning_rate_lbn,
    std::string* new_embedding_grad_lbn, std::string* state_initializer,
    OperatorConf* embedding_update_new_op_conf) {
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  const bool has_clip_grad = optimizer_conf.has_clip_conf();
  *new_embedding_grad_lbn = embedding_grad_lbn;
  std::string update_skip_if_lbn;
  std::string fuse_to_update_down_scale_by_lbn;
  double fuse_to_update_scale = 1.0;
  ScaleGrad(ctx, op_graph, job_builder, embedding_parallel_conf, embedding_scope_symbol_id,
            has_clip_grad, embedding_grad_lbn, new_embedding_grad_lbn, &update_skip_if_lbn,
            &fuse_to_update_down_scale_by_lbn, &fuse_to_update_scale);

  if (IsSupportFusedUpdatePut(is_full_cache, ctx->job_desc().enable_auto_mixed_precision(),
                              optimizer_conf.has_naive_conf(), fuse_to_update_down_scale_by_lbn,
                              update_skip_if_lbn, l1, l2,
                              optimizer_conf.weight_decay_conf().weight_decay_rate())) {
    user_op::UserOpConfWrapperBuilder fused_embedding_update_put_op_builder(
        embedding_op.op_name() + "_fused_embedding_update_put" + NewUniqueId());
    user_op::UserOpConfWrapper fused_embedding_update_put_op =
        fused_embedding_update_put_op_builder.OpTypeName("one_embedding_fused_sgd_update_put")
            .Input("num_unique_ids", num_unique_ids_lbn)
            .Input("unique_ids", unique_ids_lbn)
            .Input("unique_embeddings", unique_values_lbn)
            .Input("embedding_grad", *new_embedding_grad_lbn)
            .Input("learning_rate", learning_rate_lbn)
            .Attr<double>("scale", fuse_to_update_scale)
            .Attr<std::string>("embedding_name", embedding_name)
            .Attr<int64_t>("embedding_size", embedding_size)
            .Attr<int64_t>("line_size", line_size)
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    *embedding_update_new_op_conf = fused_embedding_update_put_op.op_conf();
    if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
      embedding_update_new_op_conf->set_stream_name_hint(embedding_name + "_EMBEDDING");
    }
    return;
  }

  auto AddAdamBiasCorrectionFactorOp = [&](float beta_val,
                                           const std::string& op_name) -> std::string {
    user_op::UserOpConfWrapperBuilder op_builder(embedding_op.op_name() + op_name);
    const auto adam_bias_correction_factor_op =
        op_builder.OpTypeName("adam_bias_correction_factor")
            .Input("train_step", train_conf.train_step_lbn())
            .Attr<float>("beta", beta_val)
            .Output("out")
            .ScopeSymbolId(embedding_scope_symbol_id)
            .Build();
    job_builder->AddOps(embedding_parallel_conf, {adam_bias_correction_factor_op.op_conf()});
    return adam_bias_correction_factor_op.output("out", 0);
  };
  user_op::UserOpConfWrapperBuilder embedding_update_op_builder(
      embedding_op.op_name() + "_embedding_update" + NewUniqueId());
  std::vector<float> state_constant_init_values;
  if (optimizer_conf.has_naive_conf()) {
    embedding_update_op_builder.OpTypeName("one_embedding_sgd_update");
  } else if (optimizer_conf.has_momentum_conf()) {
    embedding_update_op_builder.OpTypeName("one_embedding_momentum_update")
        .Attr<float>("beta", optimizer_conf.momentum_conf().beta());
  } else if (optimizer_conf.has_adam_conf()) {
    const AdamModelUpdateConf& adam_conf = optimizer_conf.adam_conf();
    if (adam_conf.smart_decay()) {
      CHECK(adam_conf.do_bias_correction())
          << "when use smart decay adam, do_bias_correction should be true. but got "
          << adam_conf.do_bias_correction();
      embedding_update_op_builder.OpTypeName("one_embedding_smart_decay_sparse_adam_update")
          .Input("train_step", train_conf.train_step_lbn())
          .Attr<float>("beta1", adam_conf.beta1())
          .Attr<float>("beta2", adam_conf.beta2())
          .Attr<float>("epsilon", adam_conf.epsilon())
          .Attr<bool>("do_bias_correction", adam_conf.do_bias_correction());
    } else {
      embedding_update_op_builder.OpTypeName("one_embedding_adam_update")
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
    }
  } else if (optimizer_conf.has_adagrad_conf()) {
    const AdagradModelUpdateConf& adagrad_conf = optimizer_conf.adagrad_conf();
    state_constant_init_values.push_back(adagrad_conf.initial_accumulator_value());
    embedding_update_op_builder.OpTypeName("one_embedding_adagrad_update")
        .Input("train_step", train_conf.train_step_lbn())
        .Attr<float>("lr_decay", adagrad_conf.lr_decay())
        .Attr<float>("epsilon", adagrad_conf.epsilon());
  } else if (optimizer_conf.has_ftrl_conf()) {
    const FtrlModelUpdateConf& ftrl_conf = optimizer_conf.ftrl_conf();
    state_constant_init_values.push_back(ftrl_conf.initial_accumulator_value());
    // For `z`, its init value is 0.0.
    state_constant_init_values.push_back(0.0);
    embedding_update_op_builder.OpTypeName("one_embedding_ftrl_update")
        .Attr<float>("lr_power", ftrl_conf.lr_power())
        .Attr<float>("lambda1", ftrl_conf.lambda1())
        .Attr<float>("lambda2", ftrl_conf.lambda2())
        .Attr<float>("beta", ftrl_conf.beta());
  } else {
    UNIMPLEMENTED();
  }
  MakeConstantInitializerAttr(embedding_size, line_size, state_constant_init_values,
                              state_initializer);

  embedding_update_op_builder.Input("num_unique_ids", num_unique_ids_lbn)
      .Input("unique_embeddings", unique_values_lbn)
      .Input("learning_rate", learning_rate_lbn)
      .Attr<float>("weight_decay", optimizer_conf.weight_decay_conf().weight_decay_rate())
      .Attr<float>("l1", l1)
      .Attr<float>("l2", l2)
      .Output("updated_unique_embeddings");
  if (!update_skip_if_lbn.empty()) {
    embedding_update_op_builder.Input("skip_if", update_skip_if_lbn);
  }
  if (!fuse_to_update_down_scale_by_lbn.empty()) {
    CHECK(!has_clip_grad);
    embedding_update_op_builder.Input("down_scale_by_tensor", fuse_to_update_down_scale_by_lbn);
  }
  user_op::UserOpConfWrapper embedding_update_op =
      embedding_update_op_builder.Input("embedding_grad", *new_embedding_grad_lbn)
          .Attr<double>("scale", fuse_to_update_scale)
          .Attr<std::string>("embedding_name", embedding_name)
          .Attr<int64_t>("embedding_size", embedding_size)
          .Attr<int64_t>("line_size", line_size)
          .ScopeSymbolId(embedding_scope_symbol_id)
          .Build();
  *embedding_update_new_op_conf = embedding_update_op.op_conf();
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
    embedding_update_new_op_conf->set_stream_name_hint(embedding_name + "_EMBEDDING");
  }

  user_op::UserOpConfWrapperBuilder embedding_put_op_builder(embedding_op.op_name()
                                                             + "_embedding_put" + NewUniqueId());
  user_op::UserOpConfWrapper embedding_put_op =
      embedding_put_op_builder.OpTypeName("embedding_put")
          .Input("num_unique_ids", num_unique_ids_lbn)
          .Input("unique_ids", unique_ids_lbn)
          .Input("unique_embeddings", embedding_update_op.output("updated_unique_embeddings", 0))
          .Attr<std::string>("embedding_name", embedding_name)
          .ScopeSymbolId(embedding_scope_symbol_id)
          .Build();
  OperatorConf embedding_put_new_op_conf = embedding_put_op.op_conf();
  if (!ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_DISABLE_PIPELINED_EXECUTION", false)) {
    embedding_put_new_op_conf.set_stream_name_hint(embedding_name + "_EMBEDDING");
  }
  job_builder->AddOps(embedding_parallel_conf, {embedding_put_new_op_conf});
}

void UpdateConsumerOpConf(const OpNode* consumer, const LogicalBlobId& out,
                          const std::string& new_out_lbn,
                          HashMap<std::string, OperatorConf>* op_name2op_conf) {
  const std::string& consumer_op_name = consumer->op().op_name();
  if (op_name2op_conf->find(consumer_op_name) == op_name2op_conf->end()) {
    (*op_name2op_conf)[consumer_op_name] = consumer->op().op_conf();
  }
  for (const std::string& ibn : consumer->op().input_bns()) {
    if (consumer->op().BnInOp2Lbi(ibn) == out) {
      OperatorConf& consumer_op_conf = op_name2op_conf->at(consumer_op_name);
      const auto& new_val = new_out_lbn;
      const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_val);
      CHECK_EQ(GenLogicalBlobName(out), old_val);
    }
  }
}

std::string GlobalAbsMaxMin(JobBuilder* job_builder,
                            const HashMap<std::string, std::string>& shadow_op_name2grad_lbn,
                            float p, const std::string& total_norm_lbn, bool max_or_min,
                            const ParallelConf& embedding_parallel_conf,
                            const int64_t embedding_scope_symbol_id,
                            const ParallelConf& parallel_conf, const int64_t scope_symbol_id) {
  bool has_split = true;
  std::string multi_reduce_op_type_name =
      has_split ? (max_or_min ? "local_multi_reduce_max_abs" : "local_multi_reduce_min_abs")
                : (max_or_min ? "multi_reduce_max_abs" : "multi_reduce_min_abs");
  std::string multi_reduce_op_name =
      "OneEmbedding-ClipGradient-GlobalNorm-MultiReduceXimumAbs-" + NewUniqueId();
  auto multi_reduce_op_builder = user_op::UserOpConfWrapperBuilder(multi_reduce_op_name)
                                     .Op(multi_reduce_op_type_name)
                                     .Output("y")
                                     .ScopeSymbolId(embedding_scope_symbol_id);
  for (const auto& pair : shadow_op_name2grad_lbn) {
    const std::string& grad_lbn = pair.second;
    multi_reduce_op_builder.Input("x", grad_lbn);
  }
  auto multi_reduce_op = multi_reduce_op_builder.Build();
  job_builder->AddOps(embedding_parallel_conf, {multi_reduce_op.op_conf()});
  std::string embedding_reduce_lbn = multi_reduce_op.output("y", 0);
  if (has_split) {
    std::string group_reduce_op_type_name = max_or_min ? "reduce_max" : "reduce_min";
    std::string group_reduce_op_name =
        "OneEmbedding-ClipGradient-GlobalNorm-GroupReduceXimum-" + NewUniqueId();
    auto group_reduce_op = user_op::UserOpConfWrapperBuilder(group_reduce_op_name)
                               .Op(group_reduce_op_type_name)
                               .Input("input_tensor", multi_reduce_op.output("y", 0))
                               .Output("output_tensor")
                               .Attr("axis", std::vector<int32_t>{0})
                               .Attr("keepdims", false)
                               .ScopeSymbolId(embedding_scope_symbol_id)
                               .Build();
    job_builder->AddOps(embedding_parallel_conf, {group_reduce_op.op_conf()});
    embedding_reduce_lbn = group_reduce_op.output("output_tensor", 0);
  }
  if (!total_norm_lbn.empty()) {
    auto stack_op_builder = user_op::UserOpConfWrapperBuilder(
                                "OneEmbedding-ClipGradient-GlobalNorm-GlobalStack-" + NewUniqueId())
                                .Op("stack")
                                .Input("in", embedding_reduce_lbn)
                                .Input("in", total_norm_lbn)
                                .Output("out")
                                .Attr("axis", int64_t(0))
                                .Attr("max_dim_size", static_cast<int64_t>(2))
                                .ScopeSymbolId(scope_symbol_id);
    auto stack_op = stack_op_builder.Build();
    job_builder->AddOps(parallel_conf, {stack_op.op_conf()});
    std::string reduce_op_type_name = max_or_min ? "reduce_max" : "reduce_min";
    std::string reduce_op_name =
        "OneEmbedding-ClipGradient-GlobalNorm-GlobalReduceXimum-" + NewUniqueId();
    auto reduce_op = user_op::UserOpConfWrapperBuilder(reduce_op_name)
                         .Op(reduce_op_type_name)
                         .Input("input_tensor", stack_op.output("out", 0))
                         .Output("output_tensor")
                         .Attr("axis", std::vector<int32_t>{0})
                         .Attr("keepdims", false)
                         .ScopeSymbolId(scope_symbol_id)
                         .Build();
    job_builder->AddOps(parallel_conf, {reduce_op.op_conf()});
    return reduce_op.output("output_tensor", 0);
  } else {
    return embedding_reduce_lbn;
  }
}

std::string GlobalNorm(JobBuilder* job_builder,
                       const HashMap<std::string, std::string>& shadow_op_name2grad_lbn, float p,
                       const std::string& total_norm_lbn,
                       const ParallelConf& embedding_parallel_conf,
                       const int64_t embedding_scope_symbol_id, const ParallelConf& parallel_conf,
                       const int64_t scope_symbol_id) {
  auto multi_reduce_sum_op_builder =
      user_op::UserOpConfWrapperBuilder("OneEmbedding-ClipGradient-GlobalNorm-MultiReduceSumPowAbs-"
                                        + NewUniqueId())
          .Op("multi_reduce_sum_pow_abs")
          .Attr("p", static_cast<float>(p))
          .Output("y")
          .ScopeSymbolId(embedding_scope_symbol_id);
  for (const auto& pair : shadow_op_name2grad_lbn) {
    const std::string grad_lbn = pair.second;
    multi_reduce_sum_op_builder.Input("x", grad_lbn);
  }
  const auto multi_reduce_sum_op = multi_reduce_sum_op_builder.Build();
  job_builder->AddOps(embedding_parallel_conf, {multi_reduce_sum_op.op_conf()});
  const std::string& embedding_sum_pow_abs_lbn = multi_reduce_sum_op.output("y", 0);
  std::string global_pow_in_lbn;
  if (!total_norm_lbn.empty()) {
    auto pow_op = user_op::UserOpConfWrapperBuilder(
                      "OneEmbedding-ClipGradient-GlobalNorm-GlobalPow-" + NewUniqueId())
                      .Op("scalar_pow")
                      .Input("in", total_norm_lbn)
                      .Attr("float_operand", static_cast<double>(p))
                      .Attr("has_float_operand", true)
                      .Output("out")
                      .ScopeSymbolId(scope_symbol_id)
                      .Build();
    job_builder->AddOps(parallel_conf, {pow_op.op_conf()});
    user_op::UserOpConfWrapperBuilder add_op_builder("OneEmbedding-ClipGradient-GlobalNorm-Add-"
                                                     + NewUniqueId());
    const auto add_op = add_op_builder.Op("add_n")
                            .Input("in", embedding_sum_pow_abs_lbn)
                            .Input("in", pow_op.output("out", 0))
                            .Output("out")
                            .ScopeSymbolId(scope_symbol_id)
                            .Build();
    job_builder->AddOps(parallel_conf, {add_op.op_conf()});
    global_pow_in_lbn = add_op.output("out", 0);
  } else {
    global_pow_in_lbn = embedding_sum_pow_abs_lbn;
  }
  auto global_pow_op = user_op::UserOpConfWrapperBuilder(
                           "OneEmbedding-ClipGradient-GlobalNorm-GlobalPow-" + NewUniqueId())
                           .Op("scalar_pow")
                           .Input("in", global_pow_in_lbn)
                           .Attr("float_operand", static_cast<double>(1.0 / p))
                           .Attr("has_float_operand", true)
                           .Output("out")
                           .ScopeSymbolId(scope_symbol_id)
                           .Build();
  job_builder->AddOps(parallel_conf, {global_pow_op.op_conf()});
  return global_pow_op.output("out", 0);
}

std::string GetClampCoeff(JobBuilder* job_builder, const std::string& total_norm_lbn,
                          float max_norm, const ParallelConf& parallel_conf,
                          const int64_t scope_symbol_id) {
  auto add_eps_ops = user_op::UserOpConfWrapperBuilder(
                         "OneEmbedding-ClipGradient-GlobalNorm-AddEps-" + NewUniqueId())
                         .Op("scalar_add")
                         .Input("in", total_norm_lbn)
                         .Attr("float_operand", 1e-6)
                         .Attr("has_float_operand", true)
                         .Output("out")
                         .ScopeSymbolId(scope_symbol_id)
                         .Build();
  job_builder->AddOps(parallel_conf, {add_eps_ops.op_conf()});

  auto inv_op =
      user_op::UserOpConfWrapperBuilder("OneEmbedding-ClipGradient-GlobalNorm-Inv-" + NewUniqueId())
          .Op("reciprocal_no_nan")
          .Input("x", add_eps_ops.output("out", 0))
          .Output("y")
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder->AddOps(parallel_conf, {inv_op.op_conf()});

  auto coeff_op = user_op::UserOpConfWrapperBuilder("OneEmbedding-ClipGradient-GlobalNorm-Coeff-"
                                                    + NewUniqueId())
                      .Op("scalar_mul")
                      .Input("in", inv_op.output("y", 0))
                      .Attr("float_operand", static_cast<double>(max_norm))
                      .Attr("has_float_operand", true)
                      .Output("out")
                      .ScopeSymbolId(scope_symbol_id)
                      .Build();
  job_builder->AddOps(parallel_conf, {coeff_op.op_conf()});

  auto clamp_coeff_op = user_op::UserOpConfWrapperBuilder(
                            "OneEmbedding-ClipGradient-GlobalNorm-Clamp-" + NewUniqueId())
                            .Op("clip_by_scalar_max")
                            .Input("x", coeff_op.output("out", 0))
                            .Attr("floating_max", 1.0)
                            .Output("y")
                            .ScopeSymbolId(scope_symbol_id)
                            .Build();
  job_builder->AddOps(parallel_conf, {clamp_coeff_op.op_conf()});
  return clamp_coeff_op.output("y", 0);
}

void ClipGradByGlobalNorm(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                          const OptimizerConf& optimizer_conf,
                          const HashMap<std::string, std::string>& shadow_op_name2grad_lbn,
                          const HashMap<std::string, OperatorConf>& grad_lbn2update_op_conf,
                          const ParallelConf& embedding_parallel_conf,
                          const int64_t embedding_scope_symbol_id,
                          HashMap<std::string, OperatorConf>* op_name2op_conf) {
  const ClipByGlobalNormConf& conf = optimizer_conf.clip_conf().clip_by_global_norm();
  double norm_type = conf.norm_type();
  auto clip_by_global_norm_pass_state =
      CHECK_JUST(ctx->MutableState<ClipByGlobalNormJobPassState>("clip_by_global_norm_state"));

  const auto NewGlobalNorm = [&](const std::string& total_norm_lbn,
                                 const ParallelConf& parallel_conf,
                                 const int64_t scope_symbol_id) -> std::string {
    if (std::isinf(norm_type) && norm_type > 0) {
      return GlobalAbsMaxMin(job_builder, shadow_op_name2grad_lbn, norm_type, total_norm_lbn, true,
                             embedding_parallel_conf, embedding_scope_symbol_id, parallel_conf,
                             scope_symbol_id);
    } else if (std::isinf(norm_type) && norm_type < 0) {
      UNIMPLEMENTED()
          << "one_embedding gradient's invalid values set to 0, so not support abs_reduce_min.";
      return GlobalAbsMaxMin(job_builder, shadow_op_name2grad_lbn, norm_type, total_norm_lbn, false,
                             embedding_parallel_conf, embedding_scope_symbol_id, parallel_conf,
                             scope_symbol_id);
    } else {
      return GlobalNorm(job_builder, shadow_op_name2grad_lbn, norm_type, total_norm_lbn,
                        embedding_parallel_conf, embedding_scope_symbol_id, parallel_conf,
                        scope_symbol_id);
    }
  };
  bool has_total_norm_state = false;
  std::string variable_op_name;
  for (const auto& var_op_name : optimizer_conf.variable_op_names()) {
    if (clip_by_global_norm_pass_state->HasTotalNormState(var_op_name)) {
      has_total_norm_state = true;
      variable_op_name = var_op_name;
      break;
    }
  }
  std::string coeff_lbn;
  if (has_total_norm_state) {
    // has_total_norm_state means there are some gradients in same optimizer group with
    // embedding_grads, the total_norm_lbn is the global norm of other gradients, embedding_grads
    // need to compute global norm with total_norm_lbn and update the consumer of the
    // total_norm_lbn, no need to compute clamp coff because it has been built in autograd pass.
    const std::shared_ptr<ClipByGlobalNormJobPassState::TotalNormState>& total_norm_state =
        clip_by_global_norm_pass_state->GetTotalNormState(variable_op_name);
    const LogicalBlobId total_norm_lbi = GenLogicalBlobId(total_norm_state->total_norm_lbn());
    std::string new_total_norm_lbn =
        NewGlobalNorm(total_norm_state->total_norm_lbn(), total_norm_state->parallel_conf(),
                      total_norm_state->scope_symbol_id());
    const OpNode* total_norm_lbn_producer = op_graph.OpNode4OpName(total_norm_lbi.op_name());
    for (const OpEdge* out_edge : total_norm_lbn_producer->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      UpdateConsumerOpConf(consumer, total_norm_lbi, new_total_norm_lbn, op_name2op_conf);
    }
    total_norm_state->set_total_norm_lbn(new_total_norm_lbn);
    coeff_lbn = total_norm_state->coeff_lbn();
  } else {
    // no norm_state means there are no gradients in same optimizer group with embedding_grad,
    // embedding_grad compute the global norm and clip independently.
    const std::string& new_total_norm_lbn =
        NewGlobalNorm("", embedding_parallel_conf, embedding_scope_symbol_id);
    coeff_lbn = GetClampCoeff(job_builder, new_total_norm_lbn, conf.max_norm(),
                              embedding_parallel_conf, embedding_scope_symbol_id);
  }
  for (const auto& pair : shadow_op_name2grad_lbn) {
    const std::string& grad_lbn = pair.second;
    const auto& it = grad_lbn2update_op_conf.find(grad_lbn);
    CHECK(it != grad_lbn2update_op_conf.end());
    OperatorConf update_op_conf = it->second;
    *(*update_op_conf.mutable_user_conf()->mutable_input())["scale_by_tensor"].mutable_s() =
        StdVec2PbRpf<std::string>({coeff_lbn});
    job_builder->AddOps(embedding_parallel_conf, {update_op_conf});
  }
}

void FilterCurGradLbnAndUpdateOpConfPairs(
    const ::google::protobuf::RepeatedPtrField<std::string>& variables,
    const HashMap<std::string, std::string>& shadow_op_name2grad_lbn,
    HashMap<std::string, std::string>* cur_shadow_op_name2grad_lbn) {
  for (const std::string& variable : variables) {
    const auto& it = shadow_op_name2grad_lbn.find(variable);
    if (it != shadow_op_name2grad_lbn.end()) {
      (*cur_shadow_op_name2grad_lbn)[variable] = it->second;
    }
  }
}

void FilterEmbeddingGradients(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                              const HashMap<std::string, std::string>& shadow_op_name2grad_lbn,
                              const HashMap<std::string, OperatorConf>& grad_lbn2update_op_conf,
                              const ParallelConf& embedding_parallel_conf,
                              const int64_t embedding_scope_symbol_id,
                              HashMap<std::string, OperatorConf>* op_name2op_conf) {
  for (const auto& optimizer_conf : job_builder->job().job_conf().train_conf().optimizer_conf()) {
    HashMap<std::string, std::string> cur_shadow_op_name2grad_lbn;
    FilterCurGradLbnAndUpdateOpConfPairs(optimizer_conf.variable_op_names(),
                                         shadow_op_name2grad_lbn, &cur_shadow_op_name2grad_lbn);
    if (!optimizer_conf.has_clip_conf()) {
      for (const auto& pair : cur_shadow_op_name2grad_lbn) {
        const auto& it = grad_lbn2update_op_conf.find(pair.second);
        CHECK(it != grad_lbn2update_op_conf.end());
        job_builder->AddOps(embedding_parallel_conf, {it->second});
      }
    } else {
      ClipGradByGlobalNorm(ctx, op_graph, job_builder, optimizer_conf, cur_shadow_op_name2grad_lbn,
                           grad_lbn2update_op_conf, embedding_parallel_conf,
                           embedding_scope_symbol_id, op_name2op_conf);
    }
  }
}

bool IsRelatedOp(const OperatorConf& op) {
  return op.has_user_conf() && (op.user_conf().op_type_name() == "one_embedding_fused_lookup");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsRelatedOp);
}

}  // namespace

class ReplaceEmbeddingOps final : public JobPass {
 public:
  ReplaceEmbeddingOps() = default;
  ~ReplaceEmbeddingOps() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder, JobPassCtx* ctx) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder, ctx);
  }
};

Maybe<void> ReplaceEmbeddingOps::Apply(const OpGraph& op_graph, JobBuilder* job_builder,
                                       JobPassCtx* ctx) const {
  ParallelConf embedding_parallel_conf;
  int64_t embedding_scope_symbol_id = 0;
  HashMap<std::string, OperatorConf> op_name2op_conf;
  HashMap<std::string, std::string> shadow_op_name2grad_lbn;
  HashMap<std::string, OperatorConf> grad_lbn2update_op_conf;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (!(op_conf.user_conf().op_type_name() == "one_embedding_fused_lookup")) { return; }
    std::vector<OperatorConf> add_ops;
    std::vector<std::string> delete_op_names;
    const user_op::UserOpConfWrapper embedding_op(op_node->op().op_conf());
    const OpNode* shadow_producer =
        op_graph.OpNode4OpName(GenLogicalBlobId(embedding_op.input("shadow", 0)).op_name());
    std::string shadow_op_name;
    if (shadow_producer->op().op_conf().has_variable_conf()) {
      shadow_op_name = shadow_producer->op().op_name();
    } else if (shadow_producer->op().op_conf().has_user_conf()
               && shadow_producer->op().op_conf().user_conf().op_type_name() == "cast") {
      const user_op::UserOpConfWrapper shadow_cast_op(shadow_producer->op().op_conf());
      const OpNode* cast_producer =
          op_graph.OpNode4OpName(GenLogicalBlobId(shadow_cast_op.input("in", 0)).op_name());
      CHECK(cast_producer->op().op_conf().has_variable_conf()) << cast_producer->op().op_name();
      shadow_op_name = cast_producer->op().op_name();
      delete_op_names.push_back(shadow_cast_op.op_name());
    } else {
      UNIMPLEMENTED() << "shadow must be variable or variable and cast";
    }
    // assume all embeddings have same placement
    embedding_scope_symbol_id = embedding_op.op_conf().scope_symbol_id();
    embedding_parallel_conf = op_node->parallel_desc().parallel_conf();
    const std::string& embedding_name = embedding_op.attr<std::string>("embedding_name");
    const int64_t line_size = embedding_op.attr<int64_t>("line_size");
    const int64_t embedding_size = embedding_op.attr<int64_t>("embedding_size");
    const bool is_full_cache = embedding_op.attr<bool>("is_full_cache");
    const int64_t seed = embedding_op.attr<int64_t>("seed");
    const int64_t parallel_num = op_node->parallel_desc().parallel_num();
    const bool use_system_gather =
        (parallel_num == 1 && ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_USE_SYSTEM_GATHER", true));
    std::string new_embeddings_lbn;

    // prefetch can not exec in advance when it consume id_shuffle_copy_out, because
    // id_shuffle_copy_out's regster_num is 1. so we set id_shuffle out to
    // prefetch_num_unique_ids_lbn and prefetch consume them for pipeline.
    std::string prefetch_num_unique_ids_lbn;
    std::string prefetch_unique_ids_lbn;
    std::string prefetch_unique_table_ids_lbn;
    std::string inner_inverse_unique_partition_indices_lbn;
    std::string num_unique_ids_lbn;
    std::string unique_ids_lbn;
    std::string unique_table_ids_lbn;
    std::string inverse_indices_lbn;
    std::string num_unique_matrix_lbn;

    BuildIdShuffle(use_system_gather, embedding_name, embedding_op, &add_ops,
                   &prefetch_num_unique_ids_lbn, &prefetch_unique_ids_lbn,
                   &prefetch_unique_table_ids_lbn, &inner_inverse_unique_partition_indices_lbn,
                   &num_unique_ids_lbn, &unique_ids_lbn, &unique_table_ids_lbn,
                   &inverse_indices_lbn, &num_unique_matrix_lbn);
    const bool is_train_job = job_builder->job().job_conf().has_train_conf();
    const bool no_optimizer_states = (embedding_size == line_size);
    const bool has_embedding_prefetch = (!is_full_cache) && (is_train_job || no_optimizer_states);

    OperatorConf embedding_prefetch_op_conf;
    OperatorConf embedding_lookup_op_conf;
    // embedding lookup op
    std::string embedding_lbn, unique_values_lbn;
    BuildEmbeddingLookup(
        ctx, job_builder, embedding_size, line_size, embedding_name, seed, has_embedding_prefetch,
        embedding_parallel_conf, embedding_op, prefetch_num_unique_ids_lbn, prefetch_unique_ids_lbn,
        prefetch_unique_table_ids_lbn, num_unique_ids_lbn, unique_ids_lbn, unique_table_ids_lbn,
        &embedding_lbn, &unique_values_lbn, &embedding_prefetch_op_conf, &embedding_lookup_op_conf);

    if (use_system_gather) {
      user_op::UserOpConfWrapperBuilder gather_op_builder(embedding_op.op_name()
                                                          + "_one_embedding_gather");
      user_op::UserOpConfWrapper gather_op =
          gather_op_builder.OpTypeName("one_embedding_gather")
              .Input("in", embedding_lbn)
              .Input("indices", inverse_indices_lbn)
              .Output("out")
              .Attr<int64_t>("embedding_size", embedding_size)
              .Attr<std::string>("embedding_name", embedding_name)
              .ScopeSymbolId(embedding_scope_symbol_id)
              .Build();
      add_ops.push_back(gather_op.op_conf());
      new_embeddings_lbn = gather_op.output("out", 0);
    } else {
      // embedding shuffle op
      BuildEmbeddingShuffle(job_builder, embedding_name, embedding_size, embedding_parallel_conf,
                            embedding_op, inverse_indices_lbn,
                            inner_inverse_unique_partition_indices_lbn, num_unique_matrix_lbn,
                            embedding_lbn, &add_ops, &new_embeddings_lbn);
    }
    delete_op_names.push_back(embedding_op.op_name());

    const LogicalBlobId out = GenLogicalBlobId(embedding_op.output("embeddings", 0));
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      UpdateConsumerOpConf(consumer, out, new_embeddings_lbn, &op_name2op_conf);
    }
    std::string state_initializer;
    // find update op
    const OpNode* producer =
        op_graph.OpNode4OpName(GenLogicalBlobId(embedding_op.input("ids", 0)).op_name());
    for (OpEdge* edge : producer->out_edges()) {
      const OpNode* consumer = edge->dst_node();
      if (consumer->op().op_conf().has_user_conf()) {
        const user_op::UserOpConfWrapper update_op_conf(consumer->op().op_conf());
        if (update_op_conf.op_type_name() != "one_embedding_fused_lookup_grad") { continue; }
        if (update_op_conf.attr<std::string>("embedding_name")
            != embedding_op.attr<std::string>("embedding_name")) {
          continue;
        }
        delete_op_names.push_back(update_op_conf.op_name());

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
        CHECK_EQ(found_embedding_optimizer, true) << shadow_op_name << " has not found optimizer";

        std::string embedding_grad_lbn;
        BuildEmbeddingGradientShuffle(
            ctx, op_graph, job_builder, op_node, embedding_name, embedding_size, use_system_gather,
            embedding_parallel_conf, embedding_scope_symbol_id, embedding_op, inverse_indices_lbn,
            inner_inverse_unique_partition_indices_lbn, num_unique_matrix_lbn,
            update_op_conf.input("embedding_grad", 0), embedding_optimizer_conf.has_clip_conf(),
            &embedding_grad_lbn);

        const OpNode* shadow_node = op_graph.OpNode4OpName(shadow_op_name);
        const VariableOpConf& shadow_variable_conf = shadow_node->op().op_conf().variable_conf();
        float l1 = 0.0;
        float l2 = 0.0;
        if (shadow_variable_conf.has_regularizer()) {
          const RegularizerConf& regularizer_conf = shadow_variable_conf.regularizer();
          if (regularizer_conf.has_l1_l2_conf()) {
            l1 = regularizer_conf.l1_l2_conf().l1();
            l2 = regularizer_conf.l1_l2_conf().l2();
          }
        }
        const std::string& learning_rate_lbn = embedding_optimizer_conf.learning_rate_lbn();

        std::string new_embedding_grad_lbn;
        OperatorConf embedding_update_op_conf;
        BuildEmbeddingUpdate(ctx, op_graph, job_builder, embedding_parallel_conf,
                             embedding_scope_symbol_id, is_full_cache, embedding_size, line_size,
                             l1, l2, embedding_name, embedding_optimizer_conf, embedding_op,
                             num_unique_ids_lbn, unique_ids_lbn, unique_values_lbn,
                             embedding_grad_lbn, learning_rate_lbn, &new_embedding_grad_lbn,
                             &state_initializer, &embedding_update_op_conf);
        shadow_op_name2grad_lbn[shadow_op_name] = new_embedding_grad_lbn;
        grad_lbn2update_op_conf[new_embedding_grad_lbn] = std::move(embedding_update_op_conf);
      }
    }
    if ((state_initializer.empty()) && !no_optimizer_states) {
      CHECK(!is_train_job) << "train job must have set state initializer";
      MakeConstantInitializerAttr(embedding_size, line_size, {}, &state_initializer);
    }
    auto state_initializer_attr = ::oneflow::AttrValue();
    state_initializer_attr.set_at_string(state_initializer);
    if (has_embedding_prefetch) {
      (*(embedding_prefetch_op_conf.mutable_user_conf()->mutable_attr()))["state_initializer"] =
          state_initializer_attr;
      add_ops.push_back(embedding_prefetch_op_conf);
    }
    (*(embedding_lookup_op_conf.mutable_user_conf()->mutable_attr()))["state_initializer"] =
        state_initializer_attr;
    add_ops.push_back(embedding_lookup_op_conf);
    job_builder->DelOps(delete_op_names);
    job_builder->AddOps(embedding_parallel_conf, add_ops);
  });
  if (shadow_op_name2grad_lbn.size() > 0) {
    FilterEmbeddingGradients(ctx, op_graph, job_builder, shadow_op_name2grad_lbn,
                             grad_lbn2update_op_conf, embedding_parallel_conf,
                             embedding_scope_symbol_id, &op_name2op_conf);
    JUST(DynamicLossScaleAddGradient(ctx, op_graph, job_builder, shadow_op_name2grad_lbn,
                                     embedding_scope_symbol_id, embedding_parallel_conf));
  }
  for (const auto& pair : op_name2op_conf) { job_builder->MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceEmbeddingOps", ReplaceEmbeddingOps);

}  // namespace oneflow
