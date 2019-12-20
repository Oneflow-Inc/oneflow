#ifndef ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

class OpGraph;
class JobBuilder;

class NonDistributedOptimizerPass final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonDistributedOptimizerPass);
  NonDistributedOptimizerPass() = default;
  ~NonDistributedOptimizerPass() = default;
  bool IsEnabled() const override { return GlobalJobDesc().enable_non_distributed_optimizer(); }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_
