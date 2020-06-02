#ifndef ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
#define ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H

#include "oneflow/core/job_completer/optimizer_base.h"

namespace oneflow {

class OptimizerRegistry {
 public:
  static OptimizerBase* Lookup(std::string name);
  static Maybe<void> Register(std::string name, OptimizerBase* optimizer);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_REGISTRY_H
