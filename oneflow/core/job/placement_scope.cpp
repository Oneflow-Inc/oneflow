#include "oneflow/core/job/placement_scope.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

Maybe<Symbol<ParallelDesc>> PlacementScope::GetParallelDesc(const OperatorConf& op_conf) const {
  if (op_conf.device_tag() == "cpu" || IsCpuOnly(op_conf)) {
    return host_parallel_desc_;
  } else {
    return device_parallel_desc_;
  }
}

}
