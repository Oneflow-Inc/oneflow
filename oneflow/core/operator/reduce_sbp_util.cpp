#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

bool ReduceSbpUtil::IsReduceAxisSplitted(const SbpInferHint& ibn_hint,
                                         const HashSet<int64_t>& reduced_axes) {
  if (ibn_hint.sbp_parallel().has_split_parallel() == false) { return false; }
  if (reduced_axes.empty()) { return true; }
  return reduced_axes.find(ibn_hint.sbp_parallel().split_parallel().axis()) != reduced_axes.end();
}

std::function<bool(int32_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(const PbRf<int32_t>& axes,
                                                                        int32_t num_axes) {
  HashSet<int32_t> axes_set = {axes.begin(), axes.end()};
  return MakePredicatorIsReducedAxis(axes_set, num_axes);
}

std::function<bool(int32_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(
    const HashSet<int32_t>& axes, int32_t num_axes) {
  auto axis_set = std::make_shared<HashSet<int32_t>>(axes);
  return [axis_set](int32_t axis) -> bool { return axis_set->find(axis) != axis_set->end(); };
}

}  // namespace oneflow
