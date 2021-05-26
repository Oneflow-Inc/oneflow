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

void GetMatmulParallelDistributionSignature(bool transpose_a, bool transpose_b,
                                            ParallelDistributionSignature* signature) {
  ParallelDistribution a_distribution;
  ParallelDistribution b_distribution;
  ParallelDistribution out_distribution;
  if (transpose_a) {
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
    a_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
  } else {
    a_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    a_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
  }
  if (transpose_b) {
    b_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
  } else {
    b_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
    b_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
  }
  out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
  out_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);

  (*signature->mutable_bn_in_op2parallel_distribution())["a"] = a_distribution;
  (*signature->mutable_bn_in_op2parallel_distribution())["b"] = b_distribution;
  (*signature->mutable_bn_in_op2parallel_distribution())["out"] = out_distribution;
}

}  // namespace

class ReplaceMatmulOpPass final : public JobPass {
 public:
  ReplaceMatmulOpPass() = default;
  ~ReplaceMatmulOpPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> ReplaceMatmulOpPass::Apply(Job* job, JobPassCtx* ctx) const {
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> op_name2op_conf;
  HashMap<std::string, ParallelDistributionSignature> op_name2parallel_distribution_signature;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    const OperatorConf& op_conf = op_node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    const std::string& op_type_name = op_conf.user_conf().op_type_name();
    if (op_type_name != "matmul_ab") { return; }
    if (op_node->parallel_desc().hierarchy()->NumAxes() != 2) { return; }
    bool transpose_a = false;
    bool transpose_b = false;
    ParallelDistributionSignature signature;
    GetMatmulParallelDistributionSignature(transpose_a, transpose_b, &signature);
    op_name2parallel_distribution_signature[op_node->op().op_name()] = signature;
    OperatorConf new_op_conf = op_conf;
    new_op_conf.mutable_user_conf()->set_op_type_name("matmul");
    op_name2op_conf[op_node->op().op_name()] = new_op_conf;
  });
  for (const auto& pair : op_name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  for (const auto& pair : op_name2parallel_distribution_signature) {
    job_builder.AddParallelDistributionSignature4OpName(pair.first, pair.second);
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ReplaceMatmulOpPass", ReplaceMatmulOpPass);

}  // namespace oneflow
