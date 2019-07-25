#ifndef RADIX_SORT_UTIL_CUH_
#define RADIX_SORT_UTIL_CUH_

namespace oneflow {

class SegmentOffsetCreator final {
 public:
  SegmentOffsetCreator(int32_t num_col) : num_col_(num_col) {}
  __device__ int32_t operator()(int32_t idx) const { return idx * num_col_; }

 private:
  int32_t num_col_;
};

}  // namespace oneflow

#endif RADIX_SORT_UTIL_CUH_
