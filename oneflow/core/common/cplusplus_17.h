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
#ifndef ONEFLOW_CORE_COMMON_CPLUSPLUS_17_H_
#define ONEFLOW_CORE_COMMON_CPLUSPLUS_17_H_

#if __cplusplus < 201703L

#include <functional>
#include <numeric>

namespace std {

// a sequential version of inclusive_scan and exclusive_scan
template<class InputIt, class OutputIt>
OutputIt inclusive_scan(InputIt first, InputIt last, OutputIt d_first) {
  return partial_sum(first, last, d_first);
}

template<class InputIt, class OutputIt, class BinaryOperation>
OutputIt inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op) {
  return partial_sum(first, last, d_first, binary_op);
}

template<class InputIt, class OutputIt, class BinaryOperation, class T>
OutputIt inclusive_scan(InputIt first, InputIt last, OutputIt d_first, BinaryOperation binary_op,
                        T init) {
  // Based on https://en.cppreference.com/w/cpp/algorithm/partial_sum
  if (first == last) return d_first;

  typename std::iterator_traits<InputIt>::value_type sum = op(*first, init);
  *d_first = sum;

  while (++first != last) {
    sum = binary_op(sum, *first);
    *++d_first = sum;
  }
  return ++d_first;
}

template<class InputIt, class OutputIt, class T, class BinaryOperation>
OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init,
                        BinaryOperation binary_op) {
  if (first == last) return d_first;

  typename std::iterator_traits<InputIt>::value_type sum = init;
  *d_first = sum;

  first--;
  last--;

  while (++first != last) {
    sum = binary_op(sum, *first);
    *++d_first = sum;
  }
  return ++d_first;
}

template<class InputIt, class OutputIt, class T>
OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt d_first, T init) {
  return exclusive_scan(first, last, d_first, init, std::plus<>());
}

}  // namespace std

#endif

#endif  // ONEFLOW_CORE_COMMON_CPLUSPLUS_17_H_
