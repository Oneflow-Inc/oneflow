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

namespace {

class FuseEmbeddingShuffleInteractionPass final : public JobPass {
 public:
  FuseEmbeddingShuffleInteractionPass() = default;
  ~FuseEmbeddingShuffleInteractionPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    // if enable quantize, not support fuse kernel.
    bool enable_quantized_comm =
        ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_ENABLE_QUANTIZED_COMM", false);
    bool enable_fuse_embedding_interaction =
        ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_FUSE_EMBEDDING_INTERACTION", false);
    return (!enable_quantized_comm && enable_fuse_embedding_interaction);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseEmbeddingShuffleInteractionPass::Apply(const OpGraph& op_graph,
                                                       JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "embedding_shuffle")) { return; }
    if (op_node->out_edges().size() > 2) { return; }
    const user_op::UserOpConfWrapper embedding_shuffle_conf(op_node->op().op_conf());
    const std::string& embeddings_lbn = embedding_shuffle_conf.output("embeddings", 0);
    const std::string& indices_lbn =
        embedding_shuffle_conf.input("inverse_unique_partition_indices", 0);
    const std::string& num_unique_matrix_lbn = embedding_shuffle_conf.input("num_unique_matrix", 0);
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(embeddings_lbn)).data_type()
            != DataType::kFloat16
        || embedding_shuffle_conf.attr<int64_t>("embedding_size") % 2 != 0) {
      // only support half and embedding_size % 2 == 0 fuse, because atomicAdd half is slow.
      return;
    }
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(indices_lbn)).data_type()
        != DataType::kUInt32) {
      // only support indices with uint32_t dtype
      return;
    }
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(num_unique_matrix_lbn)).data_type()
        != DataType::kUInt32) {
      // only support num_unique with uint32_t dtype
      return;
    }
    for (const OpEdge* out_edge : op_node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      if (!consumer->op().op_conf().has_user_conf()) { return; }
      const user_op::UserOpConfWrapper consumer_op_conf(consumer->op().op_conf());
      if (!(consumer_op_conf.op_type_name() == "fused_dot_feature_interaction"
            || consumer_op_conf.op_type_name() == "fused_dot_feature_interaction_grad")) {
        return;
      }
      if (consumer_op_conf.attr<std::string>("pooling") != "none") { return; }
      int input_size = consumer_op_conf.input_size("features");
      CHECK_GT(input_size, 0) << input_size;
      if (consumer_op_conf.input("features", input_size - 1) != embeddings_lbn) {
        // only support embeddings as last feature
        return;
      }
      user_op::UserOpConfWrapperBuilder fused_op_builder(consumer_op_conf.op_name());
      const std::string& op_type_name = consumer_op_conf.op_type_name();
      fused_op_builder.OpTypeName(op_type_name)
          .Input("sparse_feature", embeddings_lbn)
          .Input("sparse_indices", indices_lbn)
          .Input("num_valid_sparse_feature", num_unique_matrix_lbn)
          .Attr<bool>("self_interaction", consumer_op_conf.attr<bool>("self_interaction"))
          .Attr<std::string>("pooling", consumer_op_conf.attr<std::string>("pooling"));
      for (int i = 0; i < input_size - 1; ++i) {
        fused_op_builder.Input("features", consumer_op_conf.input("features", i));
      }
      OperatorConf new_op_conf = consumer->op().op_conf();
      if (op_type_name == "fused_dot_feature_interaction") {
        if (consumer_op_conf.has_input("output_concat", 0)) {
          fused_op_builder.Input("output_concat", consumer_op_conf.input("output_concat", 0));
        }
        fused_op_builder.Output("out")
            .Attr<bool>("has_output_concat", consumer_op_conf.attr<bool>("has_output_concat"))
            .Attr<int32_t>("output_padding", consumer_op_conf.attr<int32_t>("output_padding"));
        *new_op_conf.mutable_user_conf() = fused_op_builder.Build().op_conf().user_conf();
      } else {
        // fused_dot_feature_interaction_grad
        fused_op_builder.Input("dy", consumer_op_conf.input("dy", 0))
            .Output("features_grad", input_size - 1)
            .Output("sparse_feature_grad")
            .Attr<int32_t>("output_concat_grad_dim",
                           consumer_op_conf.attr<int32_t>("output_concat_grad_dim"));
        if (consumer_op_conf.has_output("output_concat_grad", 0)) {
          fused_op_builder.Output("output_concat_grad");
        }
        user_op::UserOpConfWrapper fused_dot_feature_interaction_grad_op = fused_op_builder.Build();
        *new_op_conf.mutable_user_conf() =
            fused_dot_feature_interaction_grad_op.op_conf().user_conf();
        const LogicalBlobId last_feature_grad_lbi =
            GenLogicalBlobId(consumer_op_conf.output("features_grad", input_size - 1));
        std::string sparse_feature_grad_lbn =
            fused_dot_feature_interaction_grad_op.output("sparse_feature_grad", 0);
        for (const OpEdge* out_edge : consumer->out_edges()) {
          const OpNode* grad_out_node = out_edge->dst_node();
          if (out_edge->lbis().size() == 1 && out_edge->lbis().front() == last_feature_grad_lbi) {
            if (!IsUserOpWithTypeName(grad_out_node->op().op_conf(),
                                      "embedding_gradient_shuffle")) {
              return;
            }
            OperatorConf new_embedding_gradient_shuffle_conf = grad_out_node->op().op_conf();
            for (const std::string& ibn : grad_out_node->op().input_bns()) {
              if (grad_out_node->op().BnInOp2Lbi(ibn) == last_feature_grad_lbi) {
                const auto& new_val = sparse_feature_grad_lbn;
                const auto& old_val = ReplaceInputLbnInOpCustomizedConf(
                    &new_embedding_gradient_shuffle_conf, ibn, new_val);
                CHECK_EQ(GenLogicalBlobName(last_feature_grad_lbi), old_val);
              }
            }
            auto bool_attr = ::oneflow::AttrValue();
            bool_attr.set_at_bool(true);
            (*(new_embedding_gradient_shuffle_conf.mutable_user_conf()
                   ->mutable_attr()))["skip_first_scatter"] = bool_attr;
            job_builder->MutOpsOnlyOnce({new_embedding_gradient_shuffle_conf});
          }
        }
      }
      job_builder->MutOpsOnlyOnce({new_op_conf});
    }
    auto bool_attr = ::oneflow::AttrValue();
    bool_attr.set_at_bool(true);
    OperatorConf new_embedding_shuffle_conf = op_node->op().op_conf();
    (*(new_embedding_shuffle_conf.mutable_user_conf()->mutable_attr()))["skip_last_gather"] =
        bool_attr;
    job_builder->MutOpsOnlyOnce({new_embedding_shuffle_conf});
  });

  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseEmbeddingShuffleInteractionPass", FuseEmbeddingShuffleInteractionPass);

}  // namespace oneflow
