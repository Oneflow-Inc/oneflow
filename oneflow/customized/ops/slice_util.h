#ifndef ONEFLOW_CUSTOMIZED_OPS_SLICE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_OPS_SLICE_UTIL_H_

namespace oneflow {

inline int64_t RegulateSliceIndex(int64_t idx, int64_t dims) {
  idx = (idx >= 0) ? idx : idx + dims;
  idx = std::max<int64_t>(0, idx);
  idx = std::min<int64_t>(idx, dims);
  return idx;
}

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_OPS_SLICE_UTIL_H_
