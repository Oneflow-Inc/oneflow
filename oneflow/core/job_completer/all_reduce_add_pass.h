#ifndef ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_

#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

class OpGraph;

class AllReduceAddPass final : public OpGraphPass {
 public:
  AllReduceAddPass() = default;
  ~AllReduceAddPass() = default;
  bool IsEnabled() const override {
    return !GlobalJobDesc().enable_non_distributed_optimizer()
           && GlobalJobDesc().enable_all_reduce_group();
  }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ALL_REDUCE_ADD_PASS_H_
