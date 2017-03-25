#ifndef _COMMON_SPLIT_UTIL_H_
#define _COMMON_SPLIT_UTIL_H_
#include <vector>
#include <cstdint>
namespace oneflow {
void GetDimOfEachSplit(
  int64_t whole_dim, int64_t num_split, std::vector<int64_t> *split_dim);
}  // namespace oneflow
#endif  // _COMMON_SPLIT_UTIL_H_
