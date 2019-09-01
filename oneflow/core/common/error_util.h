#ifndef ONEFLOW_CORE_COMMON_ERROR_UTIL_H_
#define ONEFLOW_CORE_COMMON_ERROR_UTIL_H_

#include "oneflow/core/common/error.pb.h"

namespace oneflow {

struct ErrorUtil final {
  static Error Ok();
  static Error ProtoParseFailedError(const std::string& msg);
  static Error JobSetEmpty(const std::string& msg);
  static Error DeviceTagNotFound(const std::string& msg);

  static Error JobTypeNotSet(const std::string& msg);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_UTIL_H_
