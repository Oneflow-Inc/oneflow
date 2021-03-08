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
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

std::function<bool(const OpNode* op_node)> MakePredicatorIsSafeToDelete(const OpGraph& op_graph) {
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });
  return [=](const OpNode* op_node) {
    if (op_node->out_edges().size() > 1) { return false; }
    if (!op_node->op().op_conf().ctrl_in_op_name().empty()) { return false; }
    if (ctrl_in_op_names.find(op_node->op().op_conf().name()) != ctrl_in_op_names.end()) {
      return false;
    }
    return true;
  };
}

bool IsUserOpWithTypeName(const OperatorConf& op_conf, const std::string& op_type_name) {
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == op_type_name;
};

class FuseUpdateOpsPass final : public JobPass {
 public:
  FuseUpdateOpsPass() = default;
  ~FuseUpdateOpsPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().enable_fuse_model_update_ops();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseUpdateOpsPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  const auto IsSafeToDelete = MakePredicatorIsSafeToDelete(op_graph);
  std::vector<std::string> del_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!op_node->op().op_conf().has_user_conf()) { return; }
    const user_op::UserOpConfWrapper user_op_conf(op_node->op().op_conf());
    if (user_op_conf.op_type_name() != "sgd_update"
        && user_op_conf.op_type_name() != "momentum_update"
        && user_op_conf.op_type_name() != "adam_update"
        && user_op_conf.op_type_name() != "rmsprop_update"
        && user_op_conf.op_type_name() != "lars_update") {
      return;
    }
    if (user_op_conf.attr<double>("scale") != 1.0 || user_op_conf.attr<float>("l1") != 0.0f
        || user_op_conf.attr<float>("l2") != 0.0f) {
      return;
    }
    float l1 = 0;
    float l2 = 0;
    double scale = 1;
    bool fused = false;
    LogicalBlobId model_diff_lbi = GenLogicalBlobId(user_op_conf.input("model_diff", 0));
    std::string scale_by_tensor_lbn;

    [&]() {
      do {
        const OpNode* producer = op_graph.OpNode4OpName(model_diff_lbi.op_name());
        if (!IsUserOpWithTypeName(producer->op().op_conf(), "l1_l2_regularize_gradient")) { break; }
        if (!IsSafeToDelete(producer)) { return; }
        const user_op::UserOpConfWrapper l1_l2_regularize_gradient_op_conf(
            producer->op().op_conf());
        if (l1_l2_regularize_gradient_op_conf.input("model", 0) != user_op_conf.input("model", 0)) {
          return;
        }
        l1 = l1_l2_regularize_gradient_op_conf.attr<float>("l1");
        l2 = l1_l2_regularize_gradient_op_conf.attr<float>("l2");
        model_diff_lbi = GenLogicalBlobId(l1_l2_regularize_gradient_op_conf.input("model_diff", 0));
        del_op_names.push_back(producer->op().op_name());
        fused = true;
      } while (false);

      do {
        const OpNode* producer = op_graph.OpNode4OpName(model_diff_lbi.op_name());
        if (!IsUserOpWithTypeName(producer->op().op_conf(), "scalar_mul_by_tensor")) { break; }
        if (!IsSafeToDelete(producer)) { return; }
        const user_op::UserOpConfWrapper scalar_mul_by_tensor_op_conf(producer->op().op_conf());
        model_diff_lbi = GenLogicalBlobId(scalar_mul_by_tensor_op_conf.input("x", 0));
        scale_by_tensor_lbn = scalar_mul_by_tensor_op_conf.input("scalar", 0);
        del_op_names.push_back(producer->op().op_name());
        fused = true;
      } while (false);

      do {
        const OpNode* producer = op_graph.OpNode4OpName(model_diff_lbi.op_name());
        if (!IsUserOpWithTypeName(producer->op().op_conf(), "scalar_mul")) { break; }
        if (!IsSafeToDelete(producer)) { return; }
        const user_op::UserOpConfWrapper scalar_mul_op_conf(producer->op().op_conf());
        if (scalar_mul_op_conf.attr<bool>("has_int_operand")) {
          scale = static_cast<double>(scalar_mul_op_conf.attr<int64_t>("int_operand"));
        } else if (scalar_mul_op_conf.attr<bool>("has_float_operand")) {
          scale = scalar_mul_op_conf.attr<double>("float_operand");
        } else {
          UNIMPLEMENTED();
        }
        model_diff_lbi = GenLogicalBlobId(scalar_mul_op_conf.input("in", 0));
        del_op_names.push_back(producer->op().op_name());
        fused = true;
      } while (false);

      do {
        const OpNode* producer = op_graph.OpNode4OpName(model_diff_lbi.op_name());
        if (!IsUserOpWithTypeName(producer->op().op_conf(), "cast")) { break; }
        if (!IsSafeToDelete(producer)) { return; }
        const user_op::UserOpConfWrapper cast_op_conf(producer->op().op_conf());
        if (producer->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_op_conf.input("in", 0))).data_type()
                != DataType::kFloat16
            || cast_op_conf.attr<DataType>("dtype") != DataType::kFloat) {
          return;
        }
        model_diff_lbi = GenLogicalBlobId(cast_op_conf.input("in", 0));
        del_op_names.push_back(producer->op().op_name());
        fused = true;
      } while (false);
    }();

    if (!fused) { return; }

    user_op::UserOpConfWrapperBuilder fused_op_builder(user_op_conf.op_name());
    fused_op_builder.OpTypeName(user_op_conf.op_type_name())
        .Input("model", user_op_conf.input("model", 0))
        .Input("model_diff", GenLogicalBlobName(model_diff_lbi))
        .Input("learning_rate", user_op_conf.input("learning_rate", 0))
        .Attr<double>("scale", scale)
        .Attr<float>("l1", l1)
        .Attr<float>("l2", l2)
        .Attr<float>("weight_decay", user_op_conf.attr<float>("weight_decay"));
    if (scale_by_tensor_lbn != "") {
      fused_op_builder.Input("scale_by_tensor", scale_by_tensor_lbn);
    }
    if (user_op_conf.has_input("skip_if", 0)) {
      fused_op_builder.Input("skip_if", user_op_conf.input("skip_if", 0));
    }
    if (user_op_conf.op_type_name() == "sgd_update") {
      // do nothing
    } else if (user_op_conf.op_type_name() == "momentum_update") {
      fused_op_builder.Input("momentum", user_op_conf.input("momentum", 0))
          .Attr<float>("beta", user_op_conf.attr<float>("beta"));
    } else if (user_op_conf.op_type_name() == "adam_update") {
      fused_op_builder.Input("m", user_op_conf.input("m", 0))
          .Input("v", user_op_conf.input("v", 0))
          .Attr<float>("beta1", user_op_conf.attr<float>("beta1"))
          .Attr<float>("beta2", user_op_conf.attr<float>("beta2"))
          .Attr<float>("epsilon", user_op_conf.attr<float>("epsilon"));
    } else if (user_op_conf.op_type_name() == "rmsprop_update") {
      const bool centered = user_op_conf.attr<bool>("centered");
      fused_op_builder.Input("mean_square", user_op_conf.input("mean_square", 0.f))
          .Attr<bool>("centered", user_op_conf.attr<bool>("centered"))
          .Attr<float>("epsilon", user_op_conf.attr<float>("epsilon"))
          .Attr<float>("decay_rate", user_op_conf.attr<float>("decay_rate"));
      if (centered) {
        fused_op_builder.Input("mean_gradient", user_op_conf.input("mean_gradient", 0.f));
      }
    } else if (user_op_conf.op_type_name() == "lars_update") {
      fused_op_builder.Attr<float>("momentum_beta", user_op_conf.attr<float>("momentum_beta"))
          .Attr<float>("epsilon", user_op_conf.attr<float>("epsilon"))
          .Attr<float>("lars_coefficient", user_op_conf.attr<float>("lars_coefficient"));
    } else {
      UNIMPLEMENTED();
    }
    OperatorConf new_op_conf = user_op_conf.op_conf();
    *new_op_conf.mutable_user_conf() = fused_op_builder.Build().op_conf().user_conf();
    job_builder->MutOpsOnlyOnce({new_op_conf});
  });
  job_builder->DelOps(del_op_names);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseUpdateOpsPass", FuseUpdateOpsPass);

}  // namespace oneflow
