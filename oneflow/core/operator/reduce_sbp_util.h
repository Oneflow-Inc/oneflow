#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/sbp_infer_hint.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct ReduceSbpUtil final {
  static bool IsReduceAxisSplitted(const SbpInferHint& ibn_hint,
                                   const HashSet<int64_t>& reduced_axes);
  static std::function<bool(int32_t)> MakePredicatorIsReducedAxis(const HashSet<int32_t>& axes,
                                                                  int32_t num_axes);
  static std::function<bool(int32_t)> MakePredicatorIsReducedAxis(const PbRf<int32_t>& axes,
                                                                  int32_t num_axes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SBP_UTIL_H_
