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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/graph/op_node_kernel_reg_context.h"

namespace oneflow {

namespace {

std::string TryInsertIdentityOp(JobBuilder* job_builder, const OpGraph& op_graph,
                                const LogicalBlobId& lbi, const ParallelConf& parallel_conf) {
  const OpNode* src_node = op_graph.OpNode4OpName(lbi.op_name());
  auto identity_op = user_op::UserOpConfWrapperBuilder(lbi.op_name() + "_" + lbi.blob_name()
                                                       + "_identity_host_" + NewUniqueId())
                         .Op("identity")
                         .Input("in", GenLogicalBlobName(lbi))
                         .Output("out")
                         .ScopeSymbolId(src_node->op().op_conf().scope_symbol_id())
                         .Build();
  job_builder->AddOps(parallel_conf, {identity_op.op_conf()});

  return identity_op.output("out", 0);
}

class InsertHostIdentityOpBeforeHostInputPass final : public JobPass {
 public:
  InsertHostIdentityOpBeforeHostInputPass() = default;
  ~InsertHostIdentityOpBeforeHostInputPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> InsertHostIdentityOpBeforeHostInputPass::Apply(const OpGraph& op_graph,
                                                           JobBuilder* job_builder) const {
  op_graph.TopoForEachNode([&](const OpNode* op_node) {
    const Operator& op = op_node->op();
    if (!op.op_conf().has_user_conf()) { return; }

    const OpNodeKernelRegContext op_node_kernel_reg_ctx(op_node);

    const user_op::OpKernelRegistryResult* kernel_reg_val =
        CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
            op.op_conf().user_conf().op_type_name(), op_node_kernel_reg_ctx));
    CHECK(kernel_reg_val != nullptr)
        << "cannot find op_type: " << op.op_conf().user_conf().op_type_name()
        << " in kernel registry !";
    if (!kernel_reg_val->has_host_memory_input) { return; }

    const user_op::UserOpConfWrapper& user_op_conf_warpper = op_node_kernel_reg_ctx.user_op_conf();

    for (const auto& pair : kernel_reg_val->host_memory_inputs) {
      if (!user_op_conf_warpper.has_input(pair.first, pair.second)) { continue; }
      const LogicalBlobId& host_input_lbi =
          GenLogicalBlobId(user_op_conf_warpper.input(pair.first, pair.second));

      ParallelConf parallel_conf = op_node->parallel_desc().parallel_conf();
      parallel_conf.set_device_tag(*CHECK_JUST(DeviceTag4DeviceType(DeviceType::kCPU)));
      const std::string& new_lbn =
          TryInsertIdentityOp(job_builder, op_graph, host_input_lbi, parallel_conf);

      if (!CHECK_JUST(job_builder->IsInMutOpTransaction(op.op_name()))) {
        CHECK_JUST(job_builder->MutOpTransactionMut(op.op_conf()));
      }
      OperatorConf& mut_op_conf = CHECK_JUST(job_builder->MutOpTransactionGet(op.op_name()));
      const auto& old_val = ReplaceInputLbnInOpCustomizedConf(
          &mut_op_conf, pair.first + "_" + std::to_string(pair.second), new_lbn);
      CHECK_EQ(old_val, GenLogicalBlobName(host_input_lbi));
    }
  });
  CHECK_JUST(job_builder->MutOpTransactionCommit());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("InsertHostIdentityOpBeforeHostInputPass",
                  InsertHostIdentityOpBeforeHostInputPass);

}  // namespace oneflow
