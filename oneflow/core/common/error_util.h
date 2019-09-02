#ifndef ONEFLOW_CORE_COMMON_ERROR_UTIL_H_
#define ONEFLOW_CORE_COMMON_ERROR_UTIL_H_

#include "oneflow/core/common/error.pb.h"

namespace oneflow {

struct ErrorUtil final {
  static std::shared_ptr<ErrorProto> Ok();
  static std::shared_ptr<ErrorProto> ProtoParseFailedError(const std::string& msg);
  static std::shared_ptr<ErrorProto> JobSetEmpty(const std::string& msg);
  static std::shared_ptr<ErrorProto> DeviceTagNotFound(const std::string& msg);

  static std::shared_ptr<ErrorProto> JobTypeNotSet(const std::string& msg);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_UTIL_H_
