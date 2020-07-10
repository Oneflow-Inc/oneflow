#ifndef ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
#define ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define MATH_BINARY_ELEMENTWISE_FUNC_SEQ     \
  OF_PP_MAKE_TUPLE_SEQ("Pow", Pow)           \
  OF_PP_MAKE_TUPLE_SEQ("Atan2", Atan2)       \
  OF_PP_MAKE_TUPLE_SEQ("Floordiv", Floordiv) \
  OF_PP_MAKE_TUPLE_SEQ("Xdivy", Xdivy)       \
  OF_PP_MAKE_TUPLE_SEQ("Xlogy", Xlogy)

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
