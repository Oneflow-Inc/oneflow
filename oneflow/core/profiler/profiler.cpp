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

#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace profiler {

namespace {

bool CaseInsensitiveStringEquals(const std::string& lhs, const std::string& rhs) {
  return lhs.size() == rhs.size()
         && std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                       [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

bool StringToBool(const std::string& str) {
  return CaseInsensitiveStringEquals(str, "1") || CaseInsensitiveStringEquals(str, "true")
         || CaseInsensitiveStringEquals(str, "yes") || CaseInsensitiveStringEquals(str, "on")
         || CaseInsensitiveStringEquals(str, "y");
}

}  // namespace

void ParseBoolFlagFromEnv(const std::string& env_var, bool* flag) {
  const char* env_p = std::getenv(env_var.c_str());
  *flag = (env_p != nullptr && StringToBool(env_p));
}

}  // namespace profiler

}  // namespace oneflow
