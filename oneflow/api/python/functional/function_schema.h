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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_GENERATED_SCHEMA_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_GENERATED_SCHEMA_H_

#include <vector>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/functional/generated/functional_api.h"
#include "oneflow/api/python/functional/function_def.h"

namespace oneflow {
namespace one {
namespace functional {

struct AddSchema {
  using FType = decltype(functional::Add);
  using R = Maybe<one::Tensor>;

  static constexpr FType* func = &functional::Add;
  static constexpr size_t max_args = 2;
  static constexpr size_t max_positionals = 2;
  static constexpr size_t max_keywords = 0;

  static ReturnDef return_def;
  static std::vector<ArgumentDef> argument_def;
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_GENERATED_SCHEMA_H_
