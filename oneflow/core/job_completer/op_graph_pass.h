#ifndef ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraphPass {
 public:
  OpGraphPass() = default;
  virtual ~OpGraphPass() = default;

  void operator()(Job* job) const {
    if (IsEnabled() == false) { return; }
    Apply(job);
  }
  virtual bool IsEnabled() const { return true; }
  virtual void Apply(Job* job) const {
    const OpGraph op_graph(*job);
    Apply(op_graph, job);
  }
  virtual void Apply(const OpGraph& op_graph, Job* job) const {
    JobBuilder job_builder(job);
    Apply(op_graph, &job_builder);
  }
  virtual void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const { UNIMPLEMENTED(); }
};

#define REGISTER_FUNCTION_PASS(pass_name, pass_type) \
  COMMAND(RegisterFunctionPass(pass_name, new pass_type))

void RegisterFunctionPass(const std::string& pass_name, const OpGraphPass* pass);
bool HasFunctionPass(const std::string& pass_name);
const OpGraphPass& FunctionPass(const std::string& pass_name);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_
