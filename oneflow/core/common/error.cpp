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
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/exception.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/error_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

namespace {

void LogError(const Error& error) {
  // gdb break point
  LOG(ERROR) << error->msg();
}

std::shared_ptr<ErrorProto>* MutThreadLocalError() {
  thread_local std::shared_ptr<ErrorProto> error;
  return &error;
}

}  // namespace

Error&& Error::AddStackFrame(const std::string& file, const int64_t& line,
                             const std::string& function) {
  auto* stack_frame = error_proto_->add_stack_frame();
  stack_frame->set_file(file);
  stack_frame->set_line(line);
  stack_frame->set_function(function);
  return std::move(*this);
}

void Error::Merge(const Error& other) {
  std::string error_summary{error_proto_->error_summary()};
  std::string msg{error_proto_->msg()};
  error_proto_->MergeFrom(*other.error_proto_);
  // MergeFrom will overwrite singular field, so restore it.
  if (!error_summary.empty()) {
    error_proto_->set_error_summary(error_summary + " " + error_proto_->error_summary());
  }
  if (!msg.empty()) { error_proto_->set_msg(msg + " " + error_proto_->msg()); }
}

Error::operator std::string() const { return error_proto_->DebugString(); }

Error Error::Ok() { return std::make_shared<ErrorProto>(); }

Error Error::ProtoParseFailedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_proto_parse_failed_error();
  return error;
}

Error Error::JobSetEmptyError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_set_empty_error();
  return error;
}

Error Error::DeviceTagNotFoundError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_device_tag_not_found_error();
  return error;
}

Error Error::InvalidValueError(const std::string& error_summary) {
  auto error = std::make_shared<ErrorProto>();
  error->set_error_summary(error_summary);
  error->mutable_invalid_value_error();
  return error;
}

Error Error::IndexError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_index_error();
  return error;
}

Error Error::TypeError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_type_error();
  return error;
}

Error Error::TimeoutError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_timeout_error();
  return error;
}

Error Error::JobNameExistError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_name_exist_error();
  return error;
}

Error Error::JobNameEmptyError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_name_empty_error();
  return error;
}

Error Error::JobNameNotEqualError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_name_not_equal_error();
  return error;
}

Error Error::NoJobBuildAndInferCtxError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_no_job_build_and_infer_ctx_error();
  return error;
}

Error Error::JobConfFrozenError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_conf_frozen_error();
  return error;
}

Error Error::JobConfNotSetError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_conf_not_set_error();
  return error;
}

Error Error::JobConfRepeatedSetError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_conf_repeated_set_error();
  return error;
}

Error Error::JobTypeNotSetError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_job_type_not_set_error();
  return error;
}

Error Error::LogicalBlobNameNotExistError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_logical_blob_name_not_exist_error();
  return error;
}

Error Error::LogicalBlobNameExistError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_logical_blob_name_exist_error();
  return error;
}

Error Error::LogicalBlobNameInvalidError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_logical_blob_name_invalid_error();
  return error;
}

Error Error::OpNameExistError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_op_name_exist_error();
  return error;
}

Error Error::OpConfDeviceTagNoSetError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_op_conf_device_tag_no_set_error();
  return error;
}

Error Error::PlacementError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_placement_error();
  return error;
}

Error Error::BlobSplitAxisInferError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_blob_split_axis_infer_error();
  return error;
}

Error Error::UnknownJobBuildAndInferError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_unknown_job_build_and_infer_error();
  return error;
}

Error Error::CheckFailedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_check_failed_error();
  return error;
}

Error Error::ValueNotFoundError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_value_not_found_error();
  return error;
}

Error Error::TodoError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_todo_error();
  return error;
}

Error Error::UnimplementedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_unimplemented_error();
  return error;
}

Error Error::RuntimeError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_runtime_error();
  return error;
}

Error Error::OutOfMemoryError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_out_of_memory_error();
  return error;
}

Error Error::BoxingNotSupportedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_boxing_not_supported_error();
  return error;
}

Error Error::OpKernelNotFoundError(const std::string& error_summary,
                                   const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<ErrorProto>();
  error->set_error_summary(error_summary);
  auto* op_kernel_not_found_error = error->mutable_op_kernel_not_found_error();
  for (const auto& msg : error_msgs) {
    op_kernel_not_found_error->add_op_kernels_not_found_debug_str(msg);
  }
  return error;
}

Error Error::MultipleOpKernelsMatchedError(const std::string& error_summary,
                                           const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<ErrorProto>();
  error->set_error_summary(error_summary);
  auto* multiple_op_kernels_matched_error = error->mutable_multiple_op_kernels_matched_error();
  for (const auto& msg : error_msgs) {
    multiple_op_kernels_matched_error->add_matched_op_kernels_debug_str(msg);
  }
  return error;
}

Error Error::MemoryZoneOutOfMemoryError(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
                                        uint64_t available, const std::string& device_tag) {
  auto error = std::make_shared<ErrorProto>();
  auto* memory_zone_out_of_memory_error = error->mutable_memory_zone_out_of_memory_error();
  memory_zone_out_of_memory_error->add_machine_id(std::to_string(machine_id));
  memory_zone_out_of_memory_error->add_mem_zone_id(std::to_string(mem_zone_id));
  memory_zone_out_of_memory_error->add_device_tag(device_tag);
  memory_zone_out_of_memory_error->add_available(std::to_string(available) + " bytes");
  memory_zone_out_of_memory_error->add_required(std::to_string(calc) + " bytes");
  return error;
}

Error Error::LossBlobNotFoundError(const std::string& error_summary) {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_loss_blob_not_found_error();
  error->set_error_summary(error_summary);
  return error;
}

Error Error::RwMutexedObjectNotFoundError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_rw_mutexed_object_not_found_error();
  return error;
}

Error Error::GradientFunctionNotFoundError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_gradient_function_not_found_error();
  return error;
}

Error Error::SymbolIdUninitializedError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_symbol_id_uninitialized_error();
  return error;
}

Error Error::CompileOptionWrongError() {
  auto error = std::make_shared<ErrorProto>();
  error->mutable_compile_option_wrong_error();
  return error;
}

Error Error::InputDeviceNotMatchError() {
  auto error = std::make_shared<ErrorProto>();
  auto* input_device_not_match_error = error->mutable_input_device_not_match_error();
  input_device_not_match_error->add_info(
      std::string("Input tensors are at different devices, please try to use tensor.to or "
                  "module.to to correct it."));
  return error;
}

std::string GetStackedErrorString(const std::shared_ptr<ErrorProto>& error) {
  const auto& maybe_error = TRY(FormatErrorStr(error));
  const auto& error_str = maybe_error.GetDataAndErrorProto(error->DebugString());
  CHECK_NE(error->error_type_case(), ErrorProto::ERROR_TYPE_NOT_SET);
  return error_str.first;
}

std::string GetErrorString(const std::shared_ptr<ErrorProto>& error) {
  if (IsInDebugMode()) {
    return GetStackedErrorString(error);
  } else {
    if (error->msg().empty() && error->stack_frame().size() > 0) {
      return error->stack_frame(0).error_msg();
    } else {
      return error->msg();
    }
  }
}

void ThrowError(const std::shared_ptr<ErrorProto>& error) {
  *MutThreadLocalError() = error;
  if (error->has_runtime_error()) { throw RuntimeException(GetErrorString(error)); }
  if (error->has_type_error()) { throw TypeException(GetErrorString(error)); }
  if (error->has_index_error()) { throw IndexException(GetErrorString(error)); }
  if (error->has_unimplemented_error()) { throw NotImplementedException(GetErrorString(error)); }
  throw Exception(GetStackedErrorString(error));
}

const std::shared_ptr<ErrorProto>& ThreadLocalError() { return *MutThreadLocalError(); }

const char* kOfBugIssueUploadPrompt =
    "This is a oneflow bug, please submit issues in "
    "'https://github.com/Oneflow-Inc/oneflow/issues' include the log information of the error, the "
    "minimum reproduction code, and the system information.";

}  // namespace oneflow
