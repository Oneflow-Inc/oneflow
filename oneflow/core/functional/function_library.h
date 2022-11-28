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
#include "oneflow/core/common/wrap_dim_utils.h"
#include "oneflow/core/functional/packed_functor.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/framework/tensor_methods.h"
#include "oneflow/core/common/throw.h"

namespace oneflow {
namespace one {
namespace functional {

class FunctionLibrary {
 public:
  virtual ~FunctionLibrary() = default;

  template<typename T>
  struct PackedFuncCreatorMap;

  template<typename R, typename... Args>
  struct PackedFuncCreatorMap<R(Args...)> {
    using FunctorCreator = typename std::function<PackedFunctor<R(Args...)>()>;

    static HashMap<std::string, FunctorCreator>* Get() {
      static HashMap<std::string, FunctorCreator> functors;
      return &functors;
    }
  };

  template<typename Func>
  void add_functor(const std::string& func_name, const Func& func) {
    using func_type = typename function_traits<Func>::func_type;
    add_functor_creator<Func>(
        func_name, [=]() { return PackedFunctorMaker<func_type>::make(func_name, func); });
  }

  template<typename Func>
  void add_one_functor(const std::string& func_name) {
    using func_type = typename function_traits<Func>::func_type;
    add_functor_creator<Func>(func_name, [=]() {
      // Lazily construct functor since ops maybe have not been registered.
      Func func;
      return PackedFunctorMaker<func_type>::make(func_name, func);
    });
  }

  template<typename... Fs>
  void add_functor(const std::string& func_name) {
    static_assert(sizeof...(Fs) > 0, "at least one functor is expected");
    __attribute__((__unused__)) int dummy[] = {(add_one_functor<Fs>(func_name), 0)...};
  }

  template<typename R, typename... Args>
  auto find(const std::string& func_name)
      -> Maybe<PackedFunctor<typename PackedFunctorMaker<R(Args...)>::FType>> {
    auto* functors = PackedFuncCreatorMap<typename PackedFunctorMaker<R(Args...)>::FType>::Get();
    const auto& it = functors->find(func_name);
    CHECK_OR_RETURN(it != functors->end())
        << Error::RuntimeError() << "Functor was not found for \"" << func_name
        << "\", please check whether the functor has been registered correctly or not.";
    return it->second();
  }

  static FunctionLibrary* Global() {
    static FunctionLibrary global_function_library;
    return &global_function_library;
  }

 private:
  FunctionLibrary() = default;

  template<typename Func, typename Creator>
  void add_functor_creator(const std::string& func_name, Creator creator) {
    using func_type = typename function_traits<Func>::func_type;
    auto* functors = PackedFuncCreatorMap<typename PackedFunctorMaker<func_type>::FType>::Get();
    CHECK_OR_THROW(functors->count(func_name) == 0)
        << Error::RuntimeError() << "The functor with name " << func_name
        << " has been registered more than once.";
    functors->emplace(func_name, creator);
  }
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
