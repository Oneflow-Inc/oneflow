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
#include <stdexcept>
#include "fmt/core.h"
#include "fmt/color.h"
#include "fmt/ostream.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/exception.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/error_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

StackedError::StackedError() : stack_frame_(), error_proto_(new ErrorProto()) {}

namespace {

void LogError(const Error& error) {
  // gdb break point
  LOG(ERROR) << error->msg();
}

std::shared_ptr<StackedError>* MutThreadLocalError() {
  thread_local std::shared_ptr<StackedError> error;
  return &error;
}

}  // namespace

Error&& Error::AddStackFrame(Symbol<ErrorStackFrame> error_stack_frame) {
  stacked_error_->add_stack_frame(error_stack_frame);
  return std::move(*this);
}

void Error::Merge(const Error& other) {
  auto* error_proto = stacked_error_->mut_error_proto();
  error_proto->MergeFrom(*other.stacked_error_->error_proto());
}

Error::operator std::string() const { return stacked_error_->DebugString(); }

Error Error::Ok() { return std::make_shared<StackedError>(); }

Error Error::ProtoParseFailedError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_proto_parse_failed_error();
  return error;
}

Error Error::JobSetEmptyError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_set_empty_error();
  return error;
}

Error Error::DeviceTagNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_device_tag_not_found_error();
  return error;
}

Error Error::InvalidValueError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_invalid_value_error();
  return error;
}

Error Error::IndexError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_index_error();
  return error;
}

Error Error::TypeError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_type_error();
  return error;
}

Error Error::TimeoutError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_timeout_error();
  return error;
}

Error Error::JobNameExistError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_name_exist_error();
  return error;
}

Error Error::JobNameEmptyError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_name_empty_error();
  return error;
}

Error Error::JobNameNotEqualError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_name_not_equal_error();
  return error;
}

Error Error::NoJobBuildAndInferCtxError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_no_job_build_and_infer_ctx_error();
  return error;
}

Error Error::JobConfFrozenError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_conf_frozen_error();
  return error;
}

Error Error::JobConfNotSetError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_conf_not_set_error();
  return error;
}

Error Error::JobConfRepeatedSetError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_conf_repeated_set_error();
  return error;
}

Error Error::JobTypeNotSetError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_job_type_not_set_error();
  return error;
}

Error Error::LogicalBlobNameNotExistError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_logical_blob_name_not_exist_error();
  return error;
}

Error Error::LogicalBlobNameExistError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_logical_blob_name_exist_error();
  return error;
}

Error Error::LogicalBlobNameInvalidError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_logical_blob_name_invalid_error();
  return error;
}

Error Error::OpNameExistError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_op_name_exist_error();
  return error;
}

Error Error::OpConfDeviceTagNoSetError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_op_conf_device_tag_no_set_error();
  return error;
}

Error Error::PlacementError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_placement_error();
  return error;
}

Error Error::BlobSplitAxisInferError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_blob_split_axis_infer_error();
  return error;
}

Error Error::UnknownJobBuildAndInferError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_unknown_job_build_and_infer_error();
  return error;
}

Error Error::CheckFailedError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_check_failed_error();
  return error;
}

Error Error::ValueNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_value_not_found_error();
  return error;
}

Error Error::TodoError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_todo_error();
  return error;
}

Error Error::UnimplementedError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_unimplemented_error();
  return error;
}

Error Error::RuntimeError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_runtime_error();
  return error;
}

Error Error::OutOfMemoryError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_out_of_memory_error();
  return error;
}

Error Error::BoxingNotSupportedError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_boxing_not_supported_error();
  return error;
}

Error Error::OpKernelNotFoundError(const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<StackedError>();
  auto* op_kernel_not_found_error = error->mut_error_proto()->mutable_op_kernel_not_found_error();
  for (const auto& msg : error_msgs) {
    op_kernel_not_found_error->add_op_kernels_not_found_debug_str(msg);
  }
  return error;
}

Error Error::MultipleOpKernelsMatchedError(const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<StackedError>();
  auto* multiple_op_kernels_matched_error =
      error->mut_error_proto()->mutable_multiple_op_kernels_matched_error();
  for (const auto& msg : error_msgs) {
    multiple_op_kernels_matched_error->add_matched_op_kernels_debug_str(msg);
  }
  return error;
}

Error Error::MemoryZoneOutOfMemoryError(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
                                        uint64_t available, const std::string& device_tag) {
  auto error = std::make_shared<StackedError>();
  auto* memory_zone_out_of_memory_error =
      error->mut_error_proto()->mutable_memory_zone_out_of_memory_error();
  memory_zone_out_of_memory_error->add_machine_id(std::to_string(machine_id));
  memory_zone_out_of_memory_error->add_mem_zone_id(std::to_string(mem_zone_id));
  memory_zone_out_of_memory_error->add_device_tag(device_tag);
  memory_zone_out_of_memory_error->add_available(std::to_string(available) + " bytes");
  memory_zone_out_of_memory_error->add_required(std::to_string(calc) + " bytes");
  return error;
}

Error Error::LossBlobNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_loss_blob_not_found_error();
  return error;
}

Error Error::RwMutexedObjectNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_rw_mutexed_object_not_found_error();
  return error;
}

Error Error::GradientFunctionNotFoundError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_gradient_function_not_found_error();
  return error;
}

Error Error::SymbolIdUninitializedError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_symbol_id_uninitialized_error();
  return error;
}

Error Error::CompileOptionWrongError() {
  auto error = std::make_shared<StackedError>();
  error->mut_error_proto()->mutable_compile_option_wrong_error();
  return error;
}

Error Error::InputDeviceNotMatchError() {
  auto error = std::make_shared<StackedError>();
  auto* input_device_not_match_error =
      error->mut_error_proto()->mutable_input_device_not_match_error();
  input_device_not_match_error->add_info(
      std::string("Input tensors are at different devices, please try to use tensor.to or "
                  "module.to to correct it."));
  return error;
}

std::string GetStackedErrorString(const std::shared_ptr<StackedError>& error) {
  const auto& maybe_error = TRY(FormatErrorStr(error));
  const auto& error_str = maybe_error.GetDataAndStackedError(error->DebugString());
  CHECK_NE(error->error_proto()->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);
  return error_str.first;
}

std::string GetErrorString(const std::shared_ptr<StackedError>& error) {
  std::string error_str;
  if (IsInDebugMode()) {
    error_str = GetStackedErrorString(error);
  } else {
    error_str = error->error_proto()->msg();
  }
  if (error_str.empty()) { error_str = "<No error message>"; }
  return error_str;
}

void ThrowError(const std::shared_ptr<StackedError>& error) {
  std::string error_str;
  fmt::format_to(std::back_inserter(error_str), "{}: {}\n",
                 fmt::styled("Error", fmt::emphasis::bold | fmt::fg(fmt::color::red)),
                 GetErrorString(error));
  // Append foreign stack trace (e.g. Python stack trace) when it is available.
  if (ForeignFrameThreadLocalGuard::Current().has_value()) {
    auto frame = *CHECK_JUST(ForeignFrameThreadLocalGuard::Current());
    if (!IsMainThread()) {
      if (auto* stack_getter = Singleton<ForeignStackGetter>::Get()) {
        fmt::format_to(std::back_inserter(error_str),
                       fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange),
                       "Related Python stack trace:");
        if (IsPythonStackGetterEnabledByDebugBuild()) {
          fmt::format_to(
              std::back_inserter(error_str),
              " (You are seeing this stack trace because you compiled OneFlow with "
              "CMAKE_BUILD_TYPE=Debug. If you want to see it even with other CMAKE_BUILD_TYPEs, "
              "you can set ONEFLOW_DEBUG or ONEFLOW_PYTHON_STACK_GETTER to 1)");
        }
        fmt::format_to(std::back_inserter(error_str), "\n{}",
                       stack_getter->GetFormattedStack(frame));
      } else {
        fmt::format_to(
            std::back_inserter(error_str),
            "You can set {} or {} to 1 to get the Python stack of the error.",
            fmt::styled("ONEFLOW_DEBUG", fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange)),
            fmt::styled("ONEFLOW_PYTHON_STACK_GETTER",
                        fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange)));
      }
    }
  }
  *MutThreadLocalError() = error;
  if ((*error)->has_runtime_error()) { throw RuntimeException(error_str); }
  if ((*error)->has_type_error()) { throw TypeException(error_str); }
  if ((*error)->has_index_error()) { throw IndexException(error_str); }
  if ((*error)->has_unimplemented_error()) { throw NotImplementedException(error_str); }
  throw Exception(GetStackedErrorString(error));
}

const std::shared_ptr<StackedError>& ThreadLocalError() { return *MutThreadLocalError(); }

const char* kOfBugIssueUploadPrompt = "This is a oneflow bug, please submit an issue at "
                                      "'https://github.com/Oneflow-Inc/oneflow/issues' including "
                                      "the log information of the error, the "
                                      "minimum reproduction code, and the system information.";
}  // namespace oneflow
