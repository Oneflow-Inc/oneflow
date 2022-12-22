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
#ifndef ONEFLOW_CORE_COMMON_BLAS_H_
#define ONEFLOW_CORE_COMMON_BLAS_H_

#include <type_traits>
#include <utility>
#include "oneflow/core/common/cblas.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define BLAS_NAME_SEQ                      \
  OF_PP_MAKE_TUPLE_SEQ(dot)                \
  OF_PP_MAKE_TUPLE_SEQ(swap)               \
  OF_PP_MAKE_TUPLE_SEQ(copy)               \
  OF_PP_MAKE_TUPLE_SEQ(axpy)               \
  OF_PP_MAKE_TUPLE_SEQ(scal)               \
  OF_PP_MAKE_TUPLE_SEQ(gemv)               \
  OF_PP_MAKE_TUPLE_SEQ(gemm)               \
  OF_PP_MAKE_TUPLE_SEQ(gemmBatched)        \
  OF_PP_MAKE_TUPLE_SEQ(gemmStridedBatched) \
  OF_PP_MAKE_TUPLE_SEQ(getrfBatched)       \
  OF_PP_MAKE_TUPLE_SEQ(getriBatched)

#define CBLAS_TEMPLATE(name)                                                                    \
  template<typename T, typename... Args>                                                        \
  auto cblas_##name(Args&&... args)                                                             \
      ->typename std::enable_if<std::is_same<T, float>::value,                                  \
                                decltype(cblas_##s##name(std::forward<Args>(args)...))>::type { \
    return cblas_##s##name(std::forward<Args>(args)...);                                        \
  }                                                                                             \
  template<typename T, typename... Args>                                                        \
  auto cblas_##name(Args&&... args)                                                             \
      ->typename std::enable_if<std::is_same<T, double>::value,                                 \
                                decltype(cblas_##d##name(std::forward<Args>(args)...))>::type { \
    return cblas_##d##name(std::forward<Args>(args)...);                                        \
  }

OF_PP_FOR_EACH_TUPLE(CBLAS_TEMPLATE, BLAS_NAME_SEQ);

#undef CBLAS_TEMPLATE

#undef BLAS_NAME_SEQ

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BLAS_H_
