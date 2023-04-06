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

#ifdef WITH_CUDA
#include <cudnn.h>
#endif  // WITH_CUDA

namespace oneflow {

namespace {

bool IsFusedBnAddReluSupported() {
#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)
  return true;
#else
  return false;
#endif
}

bool IsNormalizationAddReluOp(const OperatorConf& op) {
  return op.has_user_conf()
         && (op.user_conf().op_type_name() == "normalization_add_relu"
             || op.user_conf().op_type_name() == "normalization_add_relu_grad");
}

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), IsNormalizationAddReluOp);
}

}  // namespace

class CudnnFusedNormalizationAddReluPass final : public JobPass {
 public:
  CudnnFusedNormalizationAddReluPass() = default;
  ~CudnnFusedNormalizationAddReluPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    if (ctx.job_desc().job_conf().has_enable_cudnn_fused_normalization_add_relu()) {
      bool enabled = ctx.job_desc().job_conf().enable_cudnn_fused_normalization_add_relu();
      CHECK(!enabled || IsFusedBnAddReluSupported())
          << "Option 'enable_cudnn_fused_normalization_add_relu' is only supported when cuDNN "
             "version >= 7.4.1";
      return enabled;
    } else {
      return IsFusedBnAddReluSupported();
    }
  }
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> CudnnFusedNormalizationAddReluPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  const DataType mixed_precision_data_type = ctx->job_desc().mixed_precision_data_type();
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!IsNormalizationAddReluOp(op_conf)) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    const BlobDesc& x_desc =
        op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(user_op_conf.input("x", 0)));
    const int32_t axis = user_op_conf.attr<int32_t>("axis");
    if (x_desc.data_type() != mixed_precision_data_type) { return; }
    const Shape& x_shape = x_desc.shape();
    if (x_shape.Count(axis + 1) != 1) { return; }
    if (x_shape.At(axis) % 4 != 0) { return; }
    OperatorConf new_op_conf = op_conf;
    auto mute_attrs = new_op_conf.mutable_user_conf()->mutable_attr();
    auto training_it = mute_attrs->find("training");
    if (training_it != mute_attrs->end()) { mute_attrs->erase(training_it); }
    new_op_conf.mutable_user_conf()->set_op_type_name("cudnn_fused_" + op_type_name);
    job_builder.MutOpsOnlyOnce({new_op_conf});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("CudnnFusedNormalizationAddReluPass", CudnnFusedNormalizationAddReluPass);

}  // namespace oneflow
