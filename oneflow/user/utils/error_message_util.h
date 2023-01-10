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

#ifndef ONEFLOW_USER_UTILS_ERROR_MESSAGE_UTIL_H_
#define ONEFLOW_USER_UTILS_ERROR_MESSAGE_UTIL_H_
#include <string>

namespace oneflow {
// Problem: will it pollute the namespace of oneflow?

template<typename T1, typename T2>
std::string GetDefaultCheckEqErrorMsg(const std::string& left_name, const std::string& right_name,
                                      const T1& left_value, const T2& right_value);

template<typename T>
std::string GetDefaultCheckEqErrorMsg(const std::string& left_name, const std::string& right_name,
                                      const T& left_value);

std::string GetDefaultCheckTrueErrorMsg(const std::string& expected_fact);

}  // namespace oneflow

#endif  // ONEFLOW_USER_UTILS_POOL_UTIL_H_