#ifndef ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

class OpGraph;

class AllReduceSequencePass final : public OpGraphPass {
 public:
  AllReduceSequencePass() = default;
  ~AllReduceSequencePass() = default;
  bool IsEnabled() const override { return !GlobalJobDesc().disable_all_reduce_sequence(); }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_SEQUENCE_PASS_H_
