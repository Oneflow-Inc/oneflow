#ifndef _COMMON_STL_UTIL_H_
#define _COMMON_STL_UTIL_H_
#include <vector>
#include <algorithm>
namespace caffe {
namespace stl {
// Whether two vectors are equal
template <typename T>
bool VectorEqual(const std::vector<T>& first,
  const std::vector<T>& second) {
  if (first.size() != second.size()) return false;
  std::vector<T> sorted_first(first);
  std::vector<T> sorted_second(second);
  std::sort(sorted_first.begin(), sorted_first.end());
  std::sort(sorted_second.begin(), sorted_second.end());
  int32_t size = sorted_first.size();
  for (int32_t i = 0; i < size; ++i) {
    if (sorted_first[i] != sorted_second[i]) return false;
  }
  return true;
}
// Whether two vectors have no overlapping
template <typename T>
bool VectorNoOverlap(const std::vector<T>& first,
  const std::vector<T>& second) {
  std::unordered_set<T> first_set;
  for (auto elem : first) {
    first_set.insert(elem);
  }
  for (auto elem : second) {
    if (first_set.count(elem) > 0) return false;
  }
  return true;
}
template <typename T>
bool SetIsEqual(
  const std::unordered_set<T>& set_a,
  const std::unordered_set<T>& set_b)  {
  // For every element in set_a, it is also in set_b
  for (auto& elem_a : set_a) {
    if (set_b.count(elem_a) == 0) return false;
  }
  // For every element in set_b, it is also in set_a
  for (auto& elem_b : set_b) {
    if (set_a.count(elem_b) == 0) return false;
  }
  return true;
}

}  // namespace stl
}  // namespace vector
#endif  // _COMMON_STL_UTIL_H_