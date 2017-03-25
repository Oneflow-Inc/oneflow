#include "common/split_util.h"
#include "common/common.h"

namespace oneflow {
void GetDimOfEachSplit(
  int64_t whole_dim, int64_t num_split, std::vector<int64_t> *split_dim) {
  CHECK(num_split > 0);
  CHECK(whole_dim >= num_split);
  split_dim->clear();
  split_dim->resize(num_split);
  int64_t dim_per_split = (whole_dim + num_split - 1) / num_split;
  // split whole_dim into x parts of dim_per_split dims and 
  // (num_split - x) parts of (dim_per_split - 1) dims
  // s.t. x*dim_per_split + (split_num - x) * (dim_per_split - 1) = whole_dim
  int64_t x = whole_dim % num_split;
  x = x == 0 ? num_split : x;
  for (int32_t idx = 0; idx < num_split; ++idx) {
    if (idx < x) {
      (*split_dim)[idx] = dim_per_split;
    } else {
      (*split_dim)[idx] = dim_per_split - 1;
    }
  }
}
}  // namespace oneflow
