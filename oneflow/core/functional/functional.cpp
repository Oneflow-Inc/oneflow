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

// TODO(): Generate this file automatically.

#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functor_library.h"

namespace oneflow {
namespace one {
namespace functional {

Maybe<one::Tensor> AddScalar(const std::shared_ptr<one::Tensor>& a,
                             const std::shared_ptr<cfg::AttrValue>& scalar) {
  // "Tensor add_scalar(Tensor, Scalar)"
  static thread_local const auto& f = JUST(FunctorLibrary::Global()->find("add_scalar"));
  return f->call<Maybe<one::Tensor>>(a, scalar);
}

Maybe<one::Tensor> Add(const std::shared_ptr<one::Tensor>& a,
                       const std::shared_ptr<one::Tensor>& b) {
  // "Tensor add(Tensor, Tensor)"
  static thread_local const auto& f = JUST(FunctorLibrary::Global()->find("add"));
  return f->call<Maybe<one::Tensor>>(a, b);
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
