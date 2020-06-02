#ifndef ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H
#define ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H

#include "oneflow/core/common/util.h"

namespace oneflow {

class OptimizerBase {
 public:
  OptimizerBase() = default;
  virtual ~OptimizerBase() = default;

  virtual void Build(const std::string& handler_uuid, int64_t ofblob_ptr) const { UNIMPLEMENTED(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H
