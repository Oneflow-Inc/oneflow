#ifndef ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
#define ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

#define MATH_BINARY_ELEMENTWISE_FUNC_SEQ     \
  OF_PP_MAKE_TUPLE_SEQ("pow", Pow)           \
  OF_PP_MAKE_TUPLE_SEQ("atan2", Atan2)       \
  OF_PP_MAKE_TUPLE_SEQ("floordiv", Floordiv) \
  OF_PP_MAKE_TUPLE_SEQ("xdivy", Xdivy)       \
  OF_PP_MAKE_TUPLE_SEQ("xlogy", Xlogy)

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_OPS_MATH_BINARY_ELEMENTWISE_SEQ_H_
