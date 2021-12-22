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

namespace oneflow {

class ReplaceEmbeddingOps final : public JobPass {
 public:
  ReplaceEmbeddingOps() = default;
  ~ReplaceEmbeddingOps() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return true; }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> ReplaceEmbeddingOps::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
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
                .Input("embedding_diff",
                       embedding_gradient_shuffle_op.output("cur_rank_unique_embedding_diff", 0))
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
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceEmbeddingOps", ReplaceEmbeddingOps);

}  // namespace oneflow
