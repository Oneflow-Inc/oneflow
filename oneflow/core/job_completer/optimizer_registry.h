#ifndef ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
#define ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H

#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job_completer/optimizer_base.h"
#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

class OptimizerRegistry {
 public:
  static OptimizerBase* Lookup(const std::string& name);
  static Maybe<void> LookupAndBuild(const std::string& name, const VariableOp& var_op,
                                    const ParallelConf& parallel_conf,
                                    const LogicalBlobId& diff_lbi_of_var_out,
                                    const ::oneflow::TrainConf& train_conf);
  static Maybe<void> Register(std::string name, OptimizerBase* optimizer);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
