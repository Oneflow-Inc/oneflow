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

  std::shared_ptr<ErrorProto> error_proto() const { return error_proto_; }
  const ErrorProto* operator->() const { return error_proto_.get(); }
  ErrorProto* operator->() { return error_proto_.get(); }
  operator std::string() const;
  void Assign(const Error& other) { error_proto_ = other.error_proto_; }

  // r-value reference is used to supporting expressions like `Error().AddStackFrame("foo.cpp",
  // ,"line", "Bar") << "invalid value"` because operator<<() need r-value reference
  Error&& AddStackFrame(const std::string& file, const int64_t& line, const std::string& function);

  static Error Ok();
  static Error ProtoParseFailedError();
  static Error JobSetEmptyError();
  static Error DeviceTagNotFoundError();
  static Error InvalidValueError(const std::string& error_summary);
  static Error IndexError();
  static Error TypeError();
  static Error TimeoutError();
  static Error JobNameExistError();
  static Error JobNameEmptyError();
  static Error JobNameNotEqualError();
  static Error NoJobBuildAndInferCtxError();
  static Error JobConfFrozenError();
  static Error JobConfNotSetError();
  static Error JobConfRepeatedSetError();
  static Error JobTypeNotSetError();
  static Error LogicalBlobNameNotExistError();
  static Error LogicalBlobNameExistError();
  static Error LogicalBlobNameInvalidError();
  static Error OpNameExistError();
  static Error OpConfDeviceTagNoSetError();
  static Error PlacementError();
  static Error BlobSplitAxisInferError();
  static Error UnknownJobBuildAndInferError();
  static Error CheckFailedError();
  static Error ValueNotFoundError();
  static Error TodoError();
  static Error UnimplementedError();
  static Error RuntimeError();
  static Error BoxingNotSupportedError();
  static Error MemoryZoneOutOfMemoryError(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
                                          uint64_t available, const std::string& device_type);
  static Error OpKernelNotFoundError(const std::string& error_summary,
                                     const std::vector<std::string>& error_msgs);
  static Error MultipleOpKernelsMatchedError(const std::string& error_summary,
                                             const std::vector<std::string>& error_msgs);
  static Error LossBlobNotFoundError(const std::string& error_summary);

  static Error RwMutexedObjectNotFoundError();

  // gradient
  static Error GradientFunctionNotFoundError();

  // symbol
  static Error SymbolIdUninitializedError();

  static Error CompileOptionWrongError();

  static Error InputDeviceNotMatchError();

 private:
  std::shared_ptr<ErrorProto> error_proto_;
};

void ThrowError(const std::shared_ptr<ErrorProto>& error);
const std::shared_ptr<ErrorProto>& ThreadLocalError();

template<typename T>
Error& operator<<(Error& error, const T& x) {
  std::ostringstream ss;
  ss << x;
  if (error->stack_frame().empty()) {
    error->set_msg(error->msg() + ss.str());
  } else {
    auto* stack_frame_top = error->mutable_stack_frame(error->stack_frame_size() - 1);
    stack_frame_top->set_error_msg(stack_frame_top->error_msg() + ss.str());
  }
  return error;
}

// r-value reference is used to supporting expressions like `Error() << "invalid value"`
template<typename T>
Error&& operator<<(Error&& error, const T& x) {
  error << x;
  return std::move(error);
}

template<>
inline Error&& operator<<(Error&& error, const std::stringstream& x) {
  error << x.str();
  return std::move(error);
}

template<>
inline Error&& operator<<(Error&& error, const std::ostream& x) {
  error << x.rdbuf();
  return std::move(error);
}

template<>
inline Error&& operator<<(Error&& error, const Error& other) {
  error.Assign(other);
  return std::move(error);
}

extern const char* kOfBugIssueUploadPrompt;

}  // namespace oneflow

#define PRINT_BUG_PROMPT_AND_ABORT() LOG(FATAL) << kOfBugIssueUploadPrompt

#endif  // ONEFLOW_CORE_COMMON_ERROR_H_
