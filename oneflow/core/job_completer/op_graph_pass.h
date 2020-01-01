#ifndef ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"

namespace oneflow {

class OpGraphPass {
 public:
  void operator()(Job* job) {
    if (IsEnabled() == false) { return; }
    const OpGraph op_graph(*job);
    Apply(op_graph, job);
  }
  virtual bool IsEnabled() const { return true; }
  virtual void Apply(const OpGraph& op_graph, Job* job) {
    JobBuilder job_builder(job);
    Apply(op_graph, &job_builder);
  }
  virtual void Apply(const OpGraph& op_graph, JobBuilder* job_builder) { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_OP_GRAPH_PASS_H_
