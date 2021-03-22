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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

namespace {

class DumpVariableInfoPass final : public JobPass {
 public:
  DumpVariableInfoPass() = default;
  ~DumpVariableInfoPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return Global<ResourceDesc, ForSession>::Get()->enable_debug_mode();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> DumpVariableInfoPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  int64_t cnt = 0;
  const std::string sep = "\t";
  auto log_stream =
      TeePersistentLogStream::Create("variable_table_" + std::to_string(GlobalJobDesc().job_id()));
  (*log_stream) << "id" << sep << "name" << sep << "device_tag" << sep << "parallel_num" << sep
                << "distribute" << sep << "data_type" << sep << "shape" << sep << "elem_cnt" << sep
                << "size"
                << "\n";
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return Maybe<void>::Ok(); }
    const VariableOpConf& conf = op_conf.variable_conf();
    (*log_stream) << std::to_string(cnt);
    (*log_stream) << sep;
    (*log_stream) << op_conf.name();
    (*log_stream) << sep;
    (*log_stream) << op_conf.device_tag();
    (*log_stream) << sep;
    (*log_stream) << std::to_string(node->parallel_desc().parallel_num());
    (*log_stream) << sep;
    for (int64_t i = 0; i < conf.parallel_distribution_size(); ++i) {
      (*log_stream) << conf.parallel_distribution(i);
    }
    (*log_stream) << sep;
    (*log_stream) << DataType_Name(conf.data_type());
    (*log_stream) << sep;
    const Shape shape(conf.shape());
    (*log_stream) << shape.ToString();
    (*log_stream) << sep;
    (*log_stream) << std::to_string(shape.elem_cnt());
    (*log_stream) << sep;
    (*log_stream) << std::to_string(shape.elem_cnt() * GetSizeOfDataType(conf.data_type()));
    (*log_stream) << "\n";
    cnt += 1;
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("DumpVariableInfoPass", DumpVariableInfoPass);

}  // namespace oneflow
