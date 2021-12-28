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

namespace oneflow {

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
  srand((unsigned)time(NULL));
  int job_id = rand();
  TeePersistentLogStream::Create(StrCat("my_optimized_job", job_id))->Write(job_builder->job());
  const TrainConf& train_conf = job_builder->job().job_conf().train_conf();
  auto AddScheduleOp = [&](const OptimizerConf& optimizer_conf,
                           const std::string& op_name) -> std::string {
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
  };

  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const user_op::UserOpConfWrapper user_op_conf(op_node->op().op_conf());
    if (user_op_conf.op_type_name() != "embedding_lookup_placeholder") { return; }
    LOG(ERROR) << "user_op_conf " << user_op_conf.op_name();
    std::vector<OperatorConf> add_ops;
    std::vector<std::string> delete_op_names;

    auto AddIdentityOp = [&](std::string in_lbn) -> std::string {
      user_op::UserOpConfWrapperBuilder id_shuffle_op_builder(user_op_conf.op_name() + "_identity_"
                                                              + NewUniqueId());
      user_op::UserOpConfWrapper identity_op =
          id_shuffle_op_builder.OpTypeName("identity")
              .Input("in", in_lbn)
              .Output("out")
              .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
              .Build();
      job_builder->AddOps(op_node->parallel_desc().parallel_conf(), {identity_op.op_conf()});
      return identity_op.output("out", 0);
    };

    // id shuffle op
    user_op::UserOpConfWrapperBuilder id_shuffle_op_builder(user_op_conf.op_name() + "_id_shuffle");
    user_op::UserOpConfWrapper id_shuffle_op =
        id_shuffle_op_builder.OpTypeName("id_shuffle")
            .Input("ids", user_op_conf.input("ids", 0))
            .Output("num_unique_ids")
            .Output("ids_reverse_idx")
            .Output("cur_rank_num_unique_ids")
            .Output("cur_rank_unique_ids")
            .Output("cur_rank_reverse_idx")
            .Output("num_unique_ids_matrix")
            .Output("partition_index")
            .Attr<std::string>("partitioning", user_op_conf.attr<std::string>("partitioning"))
            .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
            .Build();
    OperatorConf id_shuffle_new_op_conf = id_shuffle_op.op_conf();
    id_shuffle_new_op_conf.set_stream_name_hint("ID_SHUFFLE");
    add_ops.push_back(id_shuffle_new_op_conf);

    const std::string unique_ids_lbn = id_shuffle_op.output("cur_rank_unique_ids", 0);
    // embedding prefetch op
    user_op::UserOpConfWrapperBuilder embedding_prefetch_op_builder(user_op_conf.op_name()
                                                                    + "_embedding_prefetch");
    user_op::UserOpConfWrapper embedding_prefetch_op =
        embedding_prefetch_op_builder.OpTypeName("embedding_prefetch")
            .Input("num_unique_ids", id_shuffle_op.output("cur_rank_num_unique_ids", 0))
            .Input("unique_ids", unique_ids_lbn)
            .Output("context")
            .Attr<std::string>("name", user_op_conf.attr<std::string>("name"))
            .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
            .Build();
    OperatorConf embedding_prefetch_new_op_conf = embedding_prefetch_op.op_conf();
    embedding_prefetch_new_op_conf.set_stream_name_hint("EMBEDDING");
    add_ops.push_back(embedding_prefetch_new_op_conf);

    // embedding lookup op
    user_op::UserOpConfWrapperBuilder embedding_lookup_op_builder(user_op_conf.op_name()
                                                                  + "_embedding_lookup");
    user_op::UserOpConfWrapper embedding_lookup_op =
        embedding_lookup_op_builder.OpTypeName("embedding_lookup")
            .Input("num_unique_ids",
                   AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
            .Input("unique_ids", AddIdentityOp(unique_ids_lbn))
            .Input("context", embedding_prefetch_op.output("context", 0))
            .Output("embeddings")
            .Output("out_context")
            .Attr<int64_t>("embedding_size", user_op_conf.attr<int64_t>("embedding_size"))
            .Attr<DataType>("dtype", user_op_conf.attr<DataType>("dtype"))
            .Attr<std::string>("name", user_op_conf.attr<std::string>("name"))
            .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
            .Build();
    OperatorConf embedding_lookup_new_op_conf = embedding_lookup_op.op_conf();
    embedding_lookup_new_op_conf.set_stream_name_hint("EMBEDDING");
    add_ops.push_back(embedding_lookup_new_op_conf);

    // embedding shuffle op
    user_op::UserOpConfWrapperBuilder embedding_shuffle_op_builder(user_op_conf.op_name());
    user_op::UserOpConfWrapper embedding_shuffle_op =
        embedding_shuffle_op_builder.OpTypeName("embedding_shuffle")
            .Input("cur_rank_embeddings", embedding_lookup_op.output("embeddings", 0))
            .Input("cur_rank_num_unique_ids",
                   AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
            .Input("cur_rank_reverse_idx",
                   AddIdentityOp(id_shuffle_op.output("cur_rank_reverse_idx", 0)))
            .Input("num_unique_ids", AddIdentityOp(id_shuffle_op.output("num_unique_ids", 0)))
            .Input("ids_reverse_idx", AddIdentityOp(id_shuffle_op.output("ids_reverse_idx", 0)))
            .Input("num_unique_ids_matrix",
                   AddIdentityOp(id_shuffle_op.output("num_unique_ids_matrix", 0)))
            .Input("partition_index", AddIdentityOp(id_shuffle_op.output("partition_index", 0)))
            .Output("embeddings")
            .Attr<int64_t>("embedding_size", user_op_conf.attr<int64_t>("embedding_size"))
            .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
            .Build();
    // add_ops.push_back(embedding_shuffle_op.op_conf());
    // delete_op_names.push_back(user_op_conf.op_name());
    job_builder->MutOpOnlyOnce(embedding_shuffle_op.op_conf());

    LogicalBlobId ids_lbi = GenLogicalBlobId(user_op_conf.input("ids", 0));
    const OpNode* producer = op_graph.OpNode4OpName(ids_lbi.op_name());
    for (OpEdge* edge : producer->out_edges()) {
      const OpNode* consumer = edge->dst_node();
      if (consumer->op().op_conf().has_user_conf()) {
        const user_op::UserOpConfWrapper update_op_conf(consumer->op().op_conf());
        if (update_op_conf.op_type_name() != "sgd_embedding_update_placeholder") { continue; }
        if (update_op_conf.attr<std::string>("name") != user_op_conf.attr<std::string>("name")) {
          continue;
        }
        delete_op_names.push_back(update_op_conf.op_name());
        // embedding_gradient_shuffle op
        user_op::UserOpConfWrapperBuilder embedding_gradient_shuffle_op_builder(
            user_op_conf.op_name() + "_embedding_gradient_shuffle");
        user_op::UserOpConfWrapper embedding_gradient_shuffle_op =
            embedding_gradient_shuffle_op_builder.OpTypeName("embedding_gradient_shuffle")
                .Input("cur_rank_num_unique_ids",
                       AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
                .Input("cur_rank_reverse_idx",
                       AddIdentityOp(id_shuffle_op.output("cur_rank_reverse_idx", 0)))
                .Input("num_unique_ids", AddIdentityOp(id_shuffle_op.output("num_unique_ids", 0)))
                .Input("ids_reverse_idx", AddIdentityOp(id_shuffle_op.output("ids_reverse_idx", 0)))
                .Input("embedding_diff", update_op_conf.input("embedding_diff", 0))
                .Input("num_unique_ids_matrix",
                       AddIdentityOp(id_shuffle_op.output("num_unique_ids_matrix", 0)))
                .Input("partition_index", AddIdentityOp(id_shuffle_op.output("partition_index", 0)))
                .Output("cur_rank_unique_embedding_diff")
                .Attr<int64_t>("embedding_size", user_op_conf.attr<int64_t>("embedding_size"))
                .ScopeSymbolId(update_op_conf.op_conf().scope_symbol_id())
                .Build();
        add_ops.push_back(embedding_gradient_shuffle_op.op_conf());

        std::string embedding_diff_lbn =
            embedding_gradient_shuffle_op.output("cur_rank_unique_embedding_diff", 0);
        // dynamic loss scale
        if (train_conf.has_dynamic_loss_scale_policy()) {
          const auto& dynamic_loss_scale_state =
              CHECK_JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));

          const LogicalBlobId count_not_finite_lbi =
              GenLogicalBlobId(dynamic_loss_scale_state.count_not_finite_lbn());
          LOG(ERROR) << "count_not_finite_lbi" << count_not_finite_lbi.op_name();
          const OpNode* identity =
              op_graph.OpNode4OpName(count_not_finite_lbi.op_name());  // identity
          const LogicalBlobId identity_in_lbi = identity->op().BnInOp2Lbi("in_0");
          const OpNode* producer =
              op_graph.OpNode4OpName(identity_in_lbi.op_name());  // parallel_cast or add

          OperatorConf last_multi_input_op_conf;
          LOG(ERROR) << "producer " << producer->op().op_conf().user_conf().op_type_name();
          if (job_builder->job().job_conf().enable_gradients_stats_aggregation()) {
            if (producer->op().op_conf().has_user_conf()
                && producer->op().op_conf().user_conf().op_type_name()
                       == "multi_count_not_finite") {
              last_multi_input_op_conf = producer->op().op_conf();
              user_op::UserOpConfWrapper multi_count_not_finite_op_conf(last_multi_input_op_conf);
              user_op::UserOpConfWrapperBuilder new_multi_count_not_finite_op_builder(
                  multi_count_not_finite_op_conf.op_name());
              new_multi_count_not_finite_op_builder.OpTypeName("multi_count_not_finite");
              for (int j = 0; j < multi_count_not_finite_op_conf.input_size("x"); ++j) {
                new_multi_count_not_finite_op_builder.Input(
                    "x", multi_count_not_finite_op_conf.input("x", j));
              }
              new_multi_count_not_finite_op_builder.Input("x", embedding_diff_lbn);
              const auto new_multi_count_not_finite_op =
                  new_multi_count_not_finite_op_builder.Output("y")
                      .ScopeSymbolId(multi_count_not_finite_op_conf.op_conf().scope_symbol_id())
                      .Build();
              // only support one embedding
              job_builder->MutOpsOnlyOnce({new_multi_count_not_finite_op.op_conf()});
            } else {
              UNIMPLEMENTED();
            }
          } else {
            const auto count_not_finite_op =
                user_op::UserOpConfWrapperBuilder("System-DynamicLossScale-CountNotFinite-"
                                                  + NewUniqueId())
                    .Op("count_not_finite")
                    .Input("x", embedding_diff_lbn)
                    .Output("y")
                    .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
                    .Build();
            add_ops.push_back(count_not_finite_op.op_conf());
            if (producer->op().op_conf().has_user_conf()
                && producer->op().op_conf().user_conf().op_type_name() == "add_n") {
              last_multi_input_op_conf = producer->op().op_conf();
            } else if (producer->op().op_conf().has_user_conf()
                       && producer->op().op_conf().user_conf().op_type_name()
                              == "hierarchical_parallel_cast") {
              const LogicalBlobId in_lbi = producer->op().BnInOp2Lbi("in_0");
              LOG(ERROR) << "in op:" << in_lbi.op_name();
              const OpNode* add_n_node =
                  op_graph.OpNode4OpName(in_lbi.op_name());  // parallel_cast or add
              CHECK(add_n_node->op().op_conf().has_user_conf()
                    && producer->op().op_conf().user_conf().op_type_name() == "add_n");
              last_multi_input_op_conf = add_n_node->op().op_conf();
            }
            LOG(ERROR) << "last_multi_input_op_conf " << last_multi_input_op_conf.DebugString();
            user_op::UserOpConfWrapper add_n_op_conf(last_multi_input_op_conf);
            user_op::UserOpConfWrapperBuilder new_add_n_op_builder(add_n_op_conf.op_name());
            new_add_n_op_builder.OpTypeName("add_n");
            for (int j = 0; j < add_n_op_conf.input_size("in"); ++j) {
              new_add_n_op_builder.Input("in", add_n_op_conf.input("in", j));
            }
            new_add_n_op_builder.Input("in", count_not_finite_op.output("y", 0));
            new_add_n_op_builder.Output("out");
            const auto new_add_n_op = new_add_n_op_builder.Output("out")
                                          .ScopeSymbolId(add_n_op_conf.op_conf().scope_symbol_id())
                                          .Build();
            // only support one embedding
            job_builder->MutOpsOnlyOnce({new_add_n_op.op_conf()});
          }
        }

        // embedding_update op
        // optimizer_conf as embedding param, now use train_conf.optimizer_conf(0) to test
        const auto& optimizer_conf = train_conf.optimizer_conf(0);
        const std::string& learning_rate_lbn =
            AddScheduleOp(optimizer_conf, "System-Train-LearningRate-Scheduler_" + NewUniqueId());
        user_op::UserOpConfWrapperBuilder sgd_embedding_update_op_builder(
            user_op_conf.op_name() + "_sgd_embedding_update");
        user_op::UserOpConfWrapper sgd_embedding_update_op =
            sgd_embedding_update_op_builder.OpTypeName("sgd_embedding_update")
                .Input("num_unique_ids",
                       AddIdentityOp(id_shuffle_op.output("cur_rank_num_unique_ids", 0)))
                .Input("unique_ids", AddIdentityOp(id_shuffle_op.output("cur_rank_unique_ids", 0)))
                .Input("context", embedding_lookup_op.output("out_context", 0))
                .Input("unique_embeddings", embedding_lookup_op.output("embeddings", 0))
                .Input("embedding_diff", embedding_diff_lbn)
                .Input("learning_rate", learning_rate_lbn)
                .Attr<int64_t>("embedding_size", user_op_conf.attr<int64_t>("embedding_size"))
                .Attr<std::string>("name", user_op_conf.attr<std::string>("name"))
                .ScopeSymbolId(user_op_conf.op_conf().scope_symbol_id())
                .Build();
        OperatorConf sgd_embedding_update_new_op_conf = sgd_embedding_update_op.op_conf();
        sgd_embedding_update_new_op_conf.set_stream_name_hint("EMBEDDING");
        add_ops.push_back(sgd_embedding_update_new_op_conf);
      }
    }
    job_builder->DelOps(delete_op_names);
    job_builder->AddOps(op_node->parallel_desc().parallel_conf(), add_ops);
    LOG(ERROR) << "xxxxxxxxxxxxx delete" << delete_op_names.size();

    // OperatorConf new_op_conf = op_conf;
    // new_op_conf.set_stream_name_hint("EMBEDDING");
  });
  srand((unsigned)time(NULL));
  job_id = rand();
  TeePersistentLogStream::Create(StrCat("after_my_optimized_job", job_id))
      ->Write(job_builder->job());
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceEmbeddingOps", ReplaceEmbeddingOps);

}  // namespace oneflow
