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

#ifndef ONEFLOW_API_CPP_TESTS_API_TEST_H_
#define ONEFLOW_API_CPP_TESTS_API_TEST_H_

#include "oneflow/api/cpp/api.h"

namespace oneflow_api {

class EnvScope {  // NOLINT
 public:
  EnvScope() { initialize(); }
  ~EnvScope() { release(); }
};

Shape RandomShape();

template<typename T>
std::vector<T> RandomData(size_t size);

std::string GetExeDir();

}  // namespace oneflow_api

#endif  // !ONEFLOW_API_CPP_TESTS_API_TEST_H_
