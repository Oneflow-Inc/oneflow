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
#include "oneflow/core/functional/packed_functor.h"

namespace oneflow {
namespace one {
namespace functional {

class FunctionLibrary {
 public:
  using FunctorCreator = std::function<Maybe<Functor>()>;

  virtual ~FunctionLibrary() = default;

  template<typename Func>
  void add_functor(const std::string& func_name) {
    CHECK_EQ(functors_.count(func_name), 0)
        << "The functor with name " << func_name << " has been registered more than once.";
    functors_.emplace(func_name, []() -> Maybe<Functor> {
      using func_type = typename function_traits<Func>::func_type;
      Func func;
      auto body = std::make_shared<FunctionBodyImpl<func_type>>(func);
      FunctionSignature signatute = detail::PackFunctionSignature<func_type>::pack();
      return std::make_shared<Functor>(body, signatute);
    });
  }

  Maybe<PackedFunctor> find(const std::string& func_name) {
    const auto& it = functors_.find(func_name);
    CHECK_OR_RETURN(it != functors_.end())
        << "Functor was not found for op " << func_name
        << ", please check whether the functor has been registered correctly or not.";
    return std::make_shared<PackedFunctor>(func_name, *(JUST(it->second())));
  }

  static FunctionLibrary* Global() {
    static FunctionLibrary global_function_library;
    return &global_function_library;
  }

 private:
  FunctionLibrary() = default;

  HashMap<std::string, FunctorCreator> functors_;
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
