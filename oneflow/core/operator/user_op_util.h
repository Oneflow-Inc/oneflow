#ifndef ONEFLOW_CORE_OPERATOR_USER_OP_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_USER_OP_UTIL_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct UserOpCtx : public OpContext {
  HashMap<std::string, std::string> mut_inplace_obn2ibn;
  HashMap<std::string, std::string> con_inplace_obn2ibn;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_USER_OP_UTIL_H_
