#include "oneflow/core/common/error.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

Error::operator std::string() const { return PbMessage2TxtString(*error_proto_); }

Error Error::Ok() { return std::make_shared<ErrorProto>(); }

Error Error::ProtoParseFailedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_proto_parse_failed_error();
  return error;
}

Error Error::JobSetEmpty() {
  auto error = std::make_shared<ErrorProto>();
  error->set_job_build_and_infer_error(JobBuildAndInferError::kJobSetEmpty);
  return error;
}

Error Error::DeviceTagNotFound() {
  auto error = std::make_shared<ErrorProto>();
  error->set_job_build_and_infer_error(JobBuildAndInferError::kDeviceTagNotFound);
  return error;
}

Error Error::JobTypeNotSet() {
  auto error = std::make_shared<ErrorProto>();
  error->set_job_build_and_infer_error(JobBuildAndInferError::kJobTypeNotSet);
  return error;
}

Error Error::CheckFailed() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_check_failed();
  return error;
}

Error Error::Todo() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_todo_error();
  return error;
}

Error Error::Unimplemented() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_unimplemented_error();
  return error;
}

Error Error::BoxingNotSupported() {
  auto error = std::make_shared<ErrorProto>();
  error->set_boxing_error(BoxingError::kNotSupported);
  return error;
}

Error&& operator<=(const std::string& log_str, Error&& error) {
  LOG(ERROR) << log_str << error->msg();
  return std::move(error);
}

}  // namespace oneflow
