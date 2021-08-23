#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_UTIL_H_

#include "oneflow/core/framework/op_interpreter/boxing/boxing_dividor.h"

namespace oneflow {

extern Maybe<BoxingDividor> (*ReplaceInDeviceType)(DeviceType device_type);
extern Maybe<BoxingDividor> (*ReplaceOutDeviceType)(DeviceType device_type);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_BOXING_DIVIDOR_UTIL_H_
