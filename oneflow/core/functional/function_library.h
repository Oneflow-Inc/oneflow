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

#ifndef ONEFLOW_CORE_FUNCTIONAL_FUNCTION_LIBRARY_H_
#define ONEFLOW_CORE_FUNCTIONAL_FUNCTION_LIBRARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/functional/functor.h"

namespace oneflow {
namespace one {
namespace functional {

class FunctionLibrary {
 public:
  virtual ~FunctionLibrary() = default;

  template<typename Func>
  void add_functor(const std::string& name) {
    Func func;
    auto packed_func = PackedFunctor::MakePackedFunctor(name, std::move(func));
    functors_.emplace(name, packed_func);
  }

  Maybe<PackedFunctor> find(const std::string& name) {
    const auto& it = functors_.find(name);
    CHECK_OR_RETURN(it != functors_.end());
    return std::make_shared<PackedFunctor>(it->second);
  }

  static FunctionLibrary* Global() {
    static FunctionLibrary global_function_library;
    return &global_function_library;
  }

 private:
  FunctionLibrary() = default;
  HashMap<std::string, PackedFunctor> functors_;
};

#define ONEFLOW_FUNCTION_LIBRARY(m) ONEFLOW_FUNCTION_LIBRARY_IMPL(m, __COUNTER__)
#define ONEFLOW_FUNCTION_LIBRARY_IMPL(m, uuid)                                  \
  static void OF_PP_CAT(_oneflow_function_library_, uuid)(FunctionLibrary & m); \
  static int OF_PP_CAT(_oneflow_function_library_dummy_, uuid) = []() {         \
    FunctionLibrary* library = FunctionLibrary::Global();                       \
    OF_PP_CAT(_oneflow_function_library_, uuid)(*library);                      \
    return 0;                                                                   \
  }();                                                                          \
  void OF_PP_CAT(_oneflow_function_library_, uuid)(FunctionLibrary & m)

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_FUNCTION_LIBRARY_H_
