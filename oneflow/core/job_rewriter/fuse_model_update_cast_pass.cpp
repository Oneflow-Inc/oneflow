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

class FuseModelUpdateCastOpsPass final : public JobPass {
 public:
  FuseModelUpdateCastOpsPass() = default;
  ~FuseModelUpdateCastOpsPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return (ctx.job_desc().enable_fused_model_update_cast()
            || ParseBooleanFromEnv("ONEFLOW_FUSE_MODEL_UPDATE_CAST", false))
           && ctx.job_desc().enable_auto_mixed_precision();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    LOG(INFO) << "Enable fuse model update cast pass. ";
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseModelUpdateCastOpsPass::Apply(const OpGraph& op_graph,
                                              JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return; }
    LogicalBlobId model_copy_lbi;

    for (OpEdge* find_cast_edge : op_node->out_edges()) {
      OpNode* find_cast_node = find_cast_edge->dst_node();
      if (!IsUserOpWithTypeName(find_cast_node->op().op_conf(), "cast")) { continue; }
      const user_op::UserOpConfWrapper cast_user_conf(find_cast_node->op().op_conf());
      if (find_cast_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.input("in", 0)))
              .data_type()
          != DataType::kFloat) {
        continue;
      }
      if (find_cast_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.output("out", 0)))
              .data_type()
          != DataType::kFloat16) {
        continue;
      }
      // Currently only support for cuda, maybe remove this limit.
      if (find_cast_node->parallel_desc().device_type() != DeviceType::kCUDA) { continue; }

      for (OpEdge* find_model_update_edge : op_node->out_edges()) {
        OpNode* find_model_update_update_node = find_model_update_edge->dst_node();
        if (!IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(), "sgd_update")
            && !IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(),
                                     "adam_update")) {
          continue;
        }

        // Currently only support for cuda, maybe remove this limit.
        if (find_model_update_update_node->parallel_desc().device_type() != DeviceType::kCUDA) {
          continue;
        }

        const user_op::UserOpConfWrapper model_update_user_conf(
            find_model_update_update_node->op().op_conf());

        // Here we find cast and model_update node, Replace cast as mutable_cast_once, and add
        // model_copy to model_update node.
        user_op::UserOpConfWrapperBuilder fused_cast_op_builder(cast_user_conf.op_name());
        fused_cast_op_builder.OpTypeName("mutable_cast_once")
            .Input("in", cast_user_conf.input("in", 0))
            .Attr<DataType>("dtype", cast_user_conf.attr<DataType>("dtype"))
            .Output("out");

        CHECK(cast_user_conf.op_conf().has_scope_symbol_id());
        fused_cast_op_builder.ScopeSymbolId(cast_user_conf.op_conf().scope_symbol_id());

        OperatorConf new_cast_op_conf = cast_user_conf.op_conf();
        *new_cast_op_conf.mutable_user_conf() = fused_cast_op_builder.Build().op_conf().user_conf();
        job_builder->MutOpsOnlyOnce({new_cast_op_conf});

        const user_op::UserOpConfWrapper new_cast_user_conf(new_cast_op_conf);
        model_copy_lbi = GenLogicalBlobId(new_cast_user_conf.output("out", 0));
        user_op::UserOpConfWrapperBuilder fused_model_update_op_builder(
            model_update_user_conf.op_name());
        if (IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(), "sgd_update")) {
          fused_model_update_op_builder.OpTypeName("sgd_update")
              .Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0))
              .Input("learning_rate", model_update_user_conf.input("learning_rate", 0))
              .Attr<double>("scale", model_update_user_conf.attr<double>("scale"))
              .Attr<float>("l1", model_update_user_conf.attr<float>("l1"))
              .Attr<float>("l2", model_update_user_conf.attr<float>("l2"))
              .Attr<float>("weight_decay", model_update_user_conf.attr<float>("weight_decay"))
              .Attr<float>("learning_rate_scale",
                           model_update_user_conf.attr<float>("learning_rate_scale"));
        } else if (IsUserOpWithTypeName(find_model_update_update_node->op().op_conf(),
                                        "adam_update")) {
          fused_model_update_op_builder.OpTypeName("adam_update")
              .Input("model", model_update_user_conf.input("model", 0))
              .Input("model_diff", model_update_user_conf.input("model_diff", 0))
              .Input("m", model_update_user_conf.input("m", 0))
              .Input("v", model_update_user_conf.input("v", 0))
              .Input("learning_rate", model_update_user_conf.input("learning_rate", 0))
              .Attr<double>("scale", model_update_user_conf.attr<double>("scale"))
              .Attr<float>("l1", model_update_user_conf.attr<float>("l1"))
              .Attr<float>("l2", model_update_user_conf.attr<float>("l2"))
              .Attr<float>("weight_decay", model_update_user_conf.attr<float>("weight_decay"))
              .Attr<float>("beta1", model_update_user_conf.attr<float>("beta1"))
              .Attr<float>("beta2", model_update_user_conf.attr<float>("beta2"))
              .Attr<float>("epsilon", model_update_user_conf.attr<float>("epsilon"))
              .Attr<bool>("amsgrad", model_update_user_conf.attr<bool>("amsgrad"))
              .Attr<bool>("do_bias_correction",
                          model_update_user_conf.attr<bool>("do_bias_correction"))
              .Attr<float>("learning_rate_scale",
                           model_update_user_conf.attr<float>("learning_rate_scale"));
          ;
          if (model_update_user_conf.attr<bool>("do_bias_correction")) {
            fused_model_update_op_builder.Input(
                "bias_correction1", model_update_user_conf.input("bias_correction1", 0));
            fused_model_update_op_builder.Input(
                "bias_correction2", model_update_user_conf.input("bias_correction2", 0));
          }
          if (model_update_user_conf.attr<bool>("amsgrad")) {
            fused_model_update_op_builder.Input("max_v", model_update_user_conf.input("max_v", 0));
          }
        } else {
          UNIMPLEMENTED() << "Need support more optimizers. ";
        }
        fused_model_update_op_builder.Input("model_copy", GenLogicalBlobName(model_copy_lbi));
        CHECK(model_update_user_conf.op_conf().has_scope_symbol_id());
        fused_model_update_op_builder.ScopeSymbolId(
            model_update_user_conf.op_conf().scope_symbol_id());

        OperatorConf new_model_update_op_conf = model_update_user_conf.op_conf();
        *new_model_update_op_conf.mutable_user_conf() =
            fused_model_update_op_builder.Build().op_conf().user_conf();
        job_builder->MutOpsOnlyOnce({new_model_update_op_conf});
        break;
      }
      break;
    }
  });
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseModelUpdateCastOpsPass", FuseModelUpdateCastOpsPass);

}  // namespace oneflow
