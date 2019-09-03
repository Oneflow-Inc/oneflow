#ifndef ONEFLOW_CORE_COMMON_ERROR_H_
#define ONEFLOW_CORE_COMMON_ERROR_H_

#include <sstream>
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
  static Error CheckFailed();
  static Error Todo();
  static Error Unimplemented();

  const std::shared_ptr<ErrorProto>& error_proto() const { return error_proto_; }
  ErrorProto* operator->() const { return error_proto_.get(); }
  operator std::string() const;

 private:
  std::shared_ptr<ErrorProto> error_proto_;
};

template<typename T>
Error&& operator<<(Error&& error, const T& x) {
  std::ostringstream ss;
  ss << x;
  error->set_msg(error->msg() + ss.str());
  return std::move(error);
}

// for LOG(ERROR)
Error&& operator<=(const std::string& log_str, Error&& error);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_H_
