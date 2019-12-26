#ifndef ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

void MultiThreadLoopInOpKernel(size_t num, std::function<void(size_t i)> Callback);

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INFER_UTIL_H_
