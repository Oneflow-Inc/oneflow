#ifndef ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_

#include <type_traits>
#include <utility>
#include "oneflow/core/blas/cblas.h"

namespace oneflow {

#define CBLAS_TEMPLATE(name)                                               \
  template<typename T, typename... Args>                                   \
  auto cblas_##name(Args&&... args)                                        \
      ->typename std::enable_if<std::is_same<T, float>::value,             \
                                decltype(cblas_##s##name(                  \
                                    std::forward<Args>(args)...))>::type { \
    return cblas_##s##name(std::forward<Args>(args)...);                   \
  }                                                                        \
  template<typename T, typename... Args>                                   \
  auto cblas_##name(Args&&... args)                                        \
      ->typename std::enable_if<std::is_same<T, double>::value,            \
                                decltype(cblas_##d##name(                  \
                                    std::forward<Args>(args)...))>::type { \
    return cblas_##d##name(std::forward<Args>(args)...);                   \
  }

CBLAS_TEMPLATE(dot);
CBLAS_TEMPLATE(swap);
CBLAS_TEMPLATE(copy);
CBLAS_TEMPLATE(axpy);
CBLAS_TEMPLATE(scal);
CBLAS_TEMPLATE(gemv);
CBLAS_TEMPLATE(gemm);

#undef CBLAS_TEMPLATE

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
