#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

struct EagerBoxingInterpreterUtil {
  static bool IsPlacementSymmetrical(Symbol<ParallelDesc> src, Symbol<ParallelDesc> dst);
  static bool IsDeviceTypeGPU(Symbol<ParallelDesc> parallel_desc);
  static bool IsBoxingS2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingS2B(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingS2P(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingP2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingP2B(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingP2P(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingB2B(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingB2S(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
  static bool IsBoxingB2P(const cfg::SbpParallel& src, const cfg::SbpParallel& dst);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_UTIL_H_
