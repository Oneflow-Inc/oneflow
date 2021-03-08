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
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/pass_util.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

namespace {

class DoParallelCastBeforeWideningTypeCast final : public JobPass {
 public:
  DoParallelCastBeforeWideningTypeCast() = default;
  ~DoParallelCastBeforeWideningTypeCast() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().do_parallel_cast_before_widening_type_cast();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> DoParallelCastBeforeWideningTypeCast::Apply(const OpGraph& op_graph,
                                                        JobBuilder* job_builder) const {
  OpConfCache op_conf_cache;
  op_graph.ForEachNode([&op_conf_cache](OpNode* parallel_cast_node) {
    // find cast_fp16_to_fp32_or_double -> parallel_cast pattern
    const OperatorConf& parallel_cast_op_conf =
        op_conf_cache.GetLatest(parallel_cast_node->op().op_conf());
    if (!(parallel_cast_op_conf.has_user_conf()
          && parallel_cast_op_conf.user_conf().op_type_name() == "parallel_cast")) {
      return;
    }
    auto* cast_node = parallel_cast_node->SoleInEdge()->src_node();
    if (cast_node->out_edges().size() != 1) { return; }
    auto cast_op_conf = op_conf_cache.GetLatest(cast_node->op().op_conf());
    if (!(cast_op_conf.has_user_conf() && cast_op_conf.user_conf().op_type_name() == "cast")) {
      return;
    }
    user_op::UserOpConfWrapper cast_conf_wrapper(cast_op_conf);
    const auto cast_in_lbi = cast_node->SoleInEdge()->lbis().front();
    const auto cast_in_dtype = cast_node->LogicalBlobDesc4Lbi(cast_in_lbi).data_type();
    const auto cast_out_dtype = cast_conf_wrapper.attr<DataType>("dtype");
    if (!(cast_in_dtype == DataType::kFloat16
          && (cast_out_dtype == DataType::kFloat || cast_out_dtype == DataType::kDouble))) {
      return;
    }

    user_op::UserOpConfWrapper parallel_cast_conf_wrapper(parallel_cast_op_conf);
    // replace parallel_cast op input with cast op input
    {
      OperatorConf new_parallel_cast_op_conf(parallel_cast_op_conf);
      const auto& cast_input = cast_conf_wrapper.input("in", 0);
      const auto& parallel_cast_input = parallel_cast_conf_wrapper.input("in", 0);
      const auto& old_val =
          ReplaceInputLbnInOpCustomizedConf(&new_parallel_cast_op_conf, "in_0", cast_input);
      CHECK_EQ(parallel_cast_input, old_val);
      op_conf_cache.Put(new_parallel_cast_op_conf);
    }
    // replace cast op input with parallel_cast op output
    {
      OperatorConf new_cast_op_conf(cast_op_conf);
      const auto& parallel_cast_output = parallel_cast_conf_wrapper.output("out", 0);
      const auto& cast_input = cast_conf_wrapper.input("in", 0);
      const auto& old_val =
          ReplaceInputLbnInOpCustomizedConf(&new_cast_op_conf, "in_0", parallel_cast_output);
      CHECK_EQ(cast_input, old_val);
      op_conf_cache.Put(new_cast_op_conf);
    }

    // update all parallel_cast op consumers
    const std::string& cast_output = cast_conf_wrapper.output("out", 0);
    for (OpEdge* edge : parallel_cast_node->out_edges()) {
      CHECK_EQ(1, edge->lbis().size());
      LogicalBlobId cur_lbi = edge->lbis().front();
      const auto lbn = GenLogicalBlobName(cur_lbi);
      CHECK_EQ(1, edge->lbi2ibns().at(cur_lbi).size());
      const std::string& dst_ibn = edge->lbi2ibns().at(cur_lbi).front();

      OpNode* dst_node = edge->dst_node();
      OperatorConf dst_op_conf = op_conf_cache.GetLatest(dst_node->op().op_conf());
      CHECK_EQ(lbn, ReplaceInputLbnInOpCustomizedConf(&dst_op_conf, dst_ibn, cast_output));
      op_conf_cache.Put(dst_op_conf);
    }
  });
  job_builder->MutOpsOnlyOnce(op_conf_cache.op_confs());
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("DoParallelCastBeforeWideningTypeCast", DoParallelCastBeforeWideningTypeCast);

}  // namespace oneflow
