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
#include "oneflow/core/common/error.cfg.h"

namespace oneflow {

class Error final {
 public:
  Error(const std::shared_ptr<cfg::ErrorProto>& error_proto) : error_proto_(error_proto) {}
  Error(const Error&) = default;
  ~Error() = default;

  std::shared_ptr<cfg::ErrorProto> error_proto() const { return error_proto_; }
  const cfg::ErrorProto* operator->() const { return error_proto_.get(); }
  cfg::ErrorProto* operator->() { return error_proto_.get(); }
  operator std::string() const;
  void Assign(const Error& other) { error_proto_ = other.error_proto_; }

  // r-value reference is used to supporting expressions like `Error().AddStackFrame("foo.cpp",
  // "Bar") << "invalid value"` because operator<<() need r-value reference
  Error&& AddStackFrame(const std::string& location, const std::string& function);

  static Error Ok();
  static Error ProtoParseFailedError();
  static Error JobSetEmptyError();
  static Error DeviceTagNotFoundError();
  static Error ValueError(const std::string& error_summary);
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
  static Error Todo();
  static Error Unimplemented();
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
  static Error GradientFunctionNotFound();

  // symbol
  static Error SymbolIdUninitialized();

  static Error CompileOptionWrong();

 private:
  std::shared_ptr<cfg::ErrorProto> error_proto_;
};

void ThrowError(const std::shared_ptr<cfg::ErrorProto>& error);
const std::shared_ptr<cfg::ErrorProto>& ThreadLocalError();

// r-value reference is used to supporting expressions like `Error() << "invalid value"`
template<typename T>
Error&& operator<<(Error&& error, const T& x) {
  std::ostringstream ss;
  ss << x;
  if (error->stack_frame().empty()) {
    error->set_msg(error->msg() + ss.str());
  } else {
    auto* stack_frame_top = error->mutable_stack_frame(error->stack_frame_size() - 1);
    stack_frame_top->set_error_msg(stack_frame_top->error_msg() + ss.str());
  }
  return std::move(error);
}

template<>
inline Error&& operator<<(Error&& error, const Error& other) {
  error.Assign(other);
  return std::move(error);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ERROR_H_
