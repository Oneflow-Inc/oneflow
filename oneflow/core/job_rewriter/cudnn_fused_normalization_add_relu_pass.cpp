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

bool IsFusedBnAddReluSupported() {
#if defined(WITH_CUDA) && (CUDNN_VERSION >= 7401)
  return true;
#else
  return false;
#endif
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
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> CudnnFusedNormalizationAddReluPass::Apply(const OpGraph& op_graph,
                                                      JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "normalization_add_relu" && op_type_name != "normalization_add_relu_grad") {
      return;
    }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    const BlobDesc& x_desc =
        op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(user_op_conf.input("x", 0)));
    const int32_t axis = user_op_conf.attr<int32_t>("axis");
    if (x_desc.data_type() != DataType::kFloat16) { return; }
    const Shape& x_shape = x_desc.shape();
    if (x_shape.Count(axis + 1) != 1) { return; }
    if (x_shape.At(axis) % 4 != 0) { return; }
    OperatorConf new_op_conf = op_conf;
    new_op_conf.mutable_user_conf()->set_op_type_name("cudnn_fused_" + op_type_name);
    job_builder->MutOpsOnlyOnce({new_op_conf});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("CudnnFusedNormalizationAddReluPass", CudnnFusedNormalizationAddReluPass);

}  // namespace oneflow
