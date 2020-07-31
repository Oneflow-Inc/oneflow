/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_ERROR_H_
#define ONEFLOW_CORE_COMMON_ERROR_H_

#include <sstream>
#include <vector>
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
  static Error BoxingNotSupported();
  static Error MemoryZoneOutOfMemory(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
                                     uint64_t available, const std::string& device_type);
  static Error OpKernelNotFoundError(const std::string& error_summary,
                                     const std::vector<std::string>& error_msgs);
  static Error MultipleOpKernelsMatchedError(const std::string& error_summary,
                                             const std::vector<std::string>& error_msgs);
  static Error LossBlobNotFoundError(const std::string& error_summary);

  // gradient
  static Error GradientFunctionNotFound();

  std::shared_ptr<ErrorProto> error_proto() const { return error_proto_; }
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

template<>
inline Error&& operator<<(Error&& error, const JobBuildAndInferError& x) {
  error->set_job_build_and_infer_error(x);
  return std::move(error);
}

// for LOG(ERROR)
Error&& operator<=(const std::pair<std::string, std::string>& loc_and_func, Error&& error);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_H_
