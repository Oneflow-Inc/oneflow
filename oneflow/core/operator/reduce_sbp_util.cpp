#include "oneflow/core/operator/reduce_sbp_util.h"

namespace oneflow {

bool ReduceSbpUtil::IsReduceAxisSplitted(const SbpInferHint& ibn_hint,
                                         const HashSet<int64_t>& reduced_axes) {
  if (ibn_hint.sbp_parallel().has_split_parallel() == false) { return false; }
  if (reduced_axes.empty()) { return true; }
  return reduced_axes.find(ibn_hint.sbp_parallel().split_parallel().axis()) != reduced_axes.end();
}

std::function<bool(int64_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(const PbRf<int32_t>& axes) {
  HashSet<int64_t> axes_set = {axes.begin(), axes.end()};
  return MakePredicatorIsReducedAxis(axes_set);
}

std::function<bool(int64_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(const AxisVector& axes) {
  auto axes_vec_ptr = std::make_shared<AxisVector>(axes);
  return [axes_vec_ptr](int64_t axis) -> bool {
    return std::find(axes_vec_ptr->begin(), axes_vec_ptr->end(), axis) != axes_vec_ptr->end();
  };
}

std::function<bool(int64_t)> ReduceSbpUtil::MakePredicatorIsReducedAxis(
    const HashSet<int64_t>& axes) {
  auto axes_set_ptr = std::make_shared<HashSet<int64_t>>(axes);
  return [axes_set_ptr](int64_t axis) -> bool {
    return axes_set_ptr->find(axis) != axes_set_ptr->end();
  };
}

}  // namespace oneflow
