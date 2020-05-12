#ifndef ONEFLOW_CORE_EAGER_EAGER_UTIL_H_
#define ONEFLOW_CORE_EAGER_EAGER_UTIL_H_

#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace eager {

Maybe<void> RunPhysicalInstruction(const std::string& instruction_list_proto_str,
                                   const std::string& eager_symbol_list_str);
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_EAGER_UTIL_H_
