#include "oneflow/core/common/error_util.h"

namespace oneflow {

Error ErrorUtil::Ok() { return Error(); }

Error ErrorUtil::ProtoParseFailedError(const std::string& msg) {
  Error error;
  error.set_msg(msg);
  error.mutable_proto_parse_failed_error();
  return error;
}

Error ErrorUtil::JobSetEmpty(const std::string& msg) {
  Error error;
  error.set_msg(msg);
  error.set_job_build_and_infer_error(JobBuildAndInferError::kJobSetEmpty);
  return error;
}

Error ErrorUtil::DeviceTagNotFound(const std::string& msg) {
  Error error;
  error.set_msg(msg);
  error.set_job_build_and_infer_error(JobBuildAndInferError::kDeviceTagNotFound);
  return error;
}

Error ErrorUtil::JobTypeNotSet(const std::string& msg) {
  Error error;
  error.set_msg(msg);
  error.set_job_build_and_infer_error(JobBuildAndInferError::kJobTypeNotSet);
  return error;
}

}  // namespace oneflow
