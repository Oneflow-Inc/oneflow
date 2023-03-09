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

template<typename T1, typename T2>
std::string GetDefaultBinaryCheckErrorMsg(const std::string& left_name,
                                          const std::string& right_name, const T1& left_value,
                                          const T2& right_value, const std::string& op) {
  return fmt::format("Expect ({0} {4} {1}), but {2} and {3} respectively.", left_name, right_name,
                     left_value, right_value, op);
}

}  // namespace oneflow