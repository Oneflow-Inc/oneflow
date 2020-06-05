#ifndef ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H
#define ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H

#include "oneflow/core/common/util.h"

namespace oneflow {

class OptimizerBase {
 public:
  OptimizerBase() = default;
  virtual ~OptimizerBase() = default;

  virtual void Build(const std::string& var_op_conf_txt, const std::string& parallel_conf_txt,
                     const std::string& diff_lbi_of_var_out_txt,
                     const std::string& train_conf_txt) const {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COMPLETER_OPTIMIZER_BASE_H
