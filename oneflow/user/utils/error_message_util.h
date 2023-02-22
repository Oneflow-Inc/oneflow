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
#pragma once
#include <string>
#include "fmt/format.h"

namespace oneflow {
// Problem: will it pollute the namespace of oneflow?

template<typename T1, typename T2>
std::string GetDefaultCheckEqErrorMsg(const std::string& left_name, const std::string& right_name,
                                      const T1& left_value, const T2& right_value) {
  return fmt::format("The {0} and {1} are expected to be equal, but {0} is {2} and {1} is {3}",
                     left_name, right_name, left_value, right_value);
}

template<typename T>
std::string GetDefaultCheckEqErrorMsg(const std::string& left_name, const std::string& right_name,
                                      const T& left_value) {
  return fmt::format("The {0} is expected to be equal with {1}, but got {2}", left_name, right_name,
                     left_value);
}
template<typename T>
std::string GetDefaultCheckTrueErrorMsg(const T& expected_fact) {
  return fmt::format("The fact '{}' is expected to be true but not.", expected_fact);
}

}  // namespace oneflow