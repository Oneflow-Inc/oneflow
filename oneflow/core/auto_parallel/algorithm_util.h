/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_ALGORITHM_UTIL_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_ALGORITHM_UTIL_H_

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

namespace oneflow {
namespace auto_parallel {

// this function is to remove the i-th element from a vector in Constant time.
// the vector should not care about ordering.
// Be more careful about this function. Make sure that the traveling order of
// the vector goes from back to front.
template<class T>
void RemoveFrom(std::vector<T>& v, int32_t i) {
  v[i] = v.back();
  v.pop_back();
}

template<class T>
void CheckAndRemoveFrom(std::vector<T>& v, T& t) {
  for (int32_t i = v.size() - 1; i >= 0; i--) {
    if (v[i] == t) {
      RemoveFrom<T>(v, i);
      break;
    }
  }
}

// Inverse function, which transfer a vector to an unordered_map.
template<class T>
void InverseFunction(const std::vector<T>& v, std::unordered_map<T, int32_t>& inverse_map) {
  inverse_map.clear();
  for (int32_t i = 0; i < v.size(); i++) { inverse_map[v[i]] = i; }
}

// When you want to sort something but you can not move any elements, use order.
// Decide the order of sorting in a list v, we have
// v[order[i]] < v[order[j]] for all i<j.
// We could define the comparison, then we have
// comp(v[order[i]], v[order[j]]) == true for all i<j.
template<class T, class Compare>
void DecideOrder(const T& v, std::vector<int32_t>& order, const Compare& comp) {
  // Initialize order
  order.resize(v.size());
  for (int32_t i = 0; i < v.size(); i++) { order[i] = i; }
  // sort
  std::sort(order.begin(), order.end(), [&](int32_t i, int32_t j) { return comp(v[i], v[j]); });
}

// Inverse function of order
// The reason why we need the inverse_order, a.k.a id2order, instead of id2value is to eliminate
// equality. For example, we have v[0] < v[1] = v[2] < v[3] We do not know v[1] is before or after
// v[2] with comp(v[1], v[2]). But if we transfer it to order order[0] < order[1] < order[2] <
// order[3] We know the strict order.
void InverseOrder(const std::vector<int32_t>& order, std::vector<int32_t>& inverse_order);

}  // namespace auto_parallel

// Ceil quotient define a division process, denoted by (/),
// which give us the maximum part of an integer division.
// For example,
// 16 (/) 4 = 4, 17 (/) 4 = 5
// 5 (/) 2 = 3, 6 (/) 2 = 3
// 17 divide by 4 give us 5, 4, 4, 4
// The normal quotient would take the smaller one 4,
// but the ceil quotient would take the larger one 5.
int64_t CeilQuotient(int64_t dividend, int64_t divisor);

static const double kFloatDeviationMinus = 0.9999999;
static const double kFloatDeviationPlus = 1.0000001;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTO_PARALLEL_ALGORITHM_UTIL_H_
