#ifndef ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_
#define ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class OpGraph;
class JobBuilder;

class NonDistributedOptimizerPass final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonDistributedOptimizerPass);
  NonDistributedOptimizerPass() = default;
  ~NonDistributedOptimizerPass() = default;

  void Apply(const OpGraph& op_graph, JobBuilder* job_builder);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_NON_DISTRIBUTED_OPTIMIZER_PASS_H_
