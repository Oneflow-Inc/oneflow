#ifndef ONEFLOW_CORE_COMMON_ERROR_H_
#define ONEFLOW_CORE_COMMON_ERROR_H_

#include "oneflow/core/common/error.pb.h"

namespace oneflow {

class Error final {
 public:
  Error(const std::shared_ptr<ErrorProto>& error_proto) : error_proto_(error_proto) {}
  Error(const Error&) = default;
  ~Error() = default;

  static Error Ok();
  static Error ProtoParseFailedError();
  static Error JobSetEmpty();
  static Error DeviceTagNotFound();
  static Error JobTypeNotSet();

  std::shared_ptr<ErrorProto> error_proto() const { return error_proto_; }
  ErrorProto* operator->() const { return error_proto_.get(); }

 private:
  std::shared_ptr<ErrorProto> error_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_H_
