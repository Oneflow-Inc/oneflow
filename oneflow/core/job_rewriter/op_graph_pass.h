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
#ifndef ONEFLOW_CORE_JOB_REWRITER_OP_GRAPH_PASS_H_
#define ONEFLOW_CORE_JOB_REWRITER_OP_GRAPH_PASS_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraphPass {
 public:
  OpGraphPass() = default;
  virtual ~OpGraphPass() = default;

  Maybe<void> operator()(Job* job) const {
    if (IsEnabled() == false) { return Maybe<void>::Ok(); }
    return Apply(job);
  }
  virtual bool IsEnabled() const { return true; }
  virtual Maybe<void> Apply(Job* job) const {
    OpGraph op_graph;
    JUST(op_graph.Init(*job));
    return Apply(op_graph, job);
  }
  virtual Maybe<void> Apply(const OpGraph& op_graph, Job* job) const {
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
  virtual Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
    UNIMPLEMENTED();
    return Maybe<void>::Ok();
  }
};

#define REGISTER_FUNCTION_PASS(pass_name, pass_type) \
  COMMAND(RegisterFunctionPass(pass_name, new pass_type))

void RegisterFunctionPass(const std::string& pass_name, const OpGraphPass* pass);
bool HasFunctionPass(const std::string& pass_name);
const OpGraphPass& FunctionPass(const std::string& pass_name);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_OP_GRAPH_PASS_H_
