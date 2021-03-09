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
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/exception.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

void LogError(const Error& error) {
  // gdb break point
  LOG(ERROR) << error->msg();
}

std::shared_ptr<cfg::ErrorProto>* MutThreadLocalError() {
  thread_local std::shared_ptr<cfg::ErrorProto> error;
  return &error;
}

}  // namespace

Error&& Error::AddStackFrame(const std::string& location, const std::string& function) {
  auto* stack_frame = error_proto_->add_stack_frame();
  stack_frame->set_location(location);
  stack_frame->set_function(function);
  return std::move(*this);
}

Error::operator std::string() const { return error_proto_->DebugString(); }

Error Error::Ok() { return std::make_shared<cfg::ErrorProto>(); }

Error Error::ProtoParseFailedError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_proto_parse_failed_error();
  return error;
}

Error Error::JobSetEmptyError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_set_empty_error();
  return error;
}

Error Error::DeviceTagNotFoundError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_device_tag_not_found_error();
  return error;
}

Error Error::ValueError(const std::string& error_summary) {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->set_error_summary(error_summary);
  error->mutable_value_error();
  return error;
}

Error Error::JobNameExistError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_name_exist_error();
  return error;
}

Error Error::JobNameEmptyError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_name_empty_error();
  return error;
}

Error Error::JobNameNotEqualError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_name_not_equal_error();
  return error;
}

Error Error::NoJobBuildAndInferCtxError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_no_job_build_and_infer_ctx_error();
  return error;
}

Error Error::JobConfFrozenError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_conf_frozen_error();
  return error;
}

Error Error::JobConfNotSetError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_conf_not_set_error();
  return error;
}

Error Error::JobConfRepeatedSetError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_conf_repeated_set_error();
  return error;
}

Error Error::JobTypeNotSetError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_job_type_not_set_error();
  return error;
}

Error Error::LogicalBlobNameNotExistError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_logical_blob_name_not_exist_error();
  return error;
}

Error Error::LogicalBlobNameExistError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_logical_blob_name_exist_error();
  return error;
}

Error Error::LogicalBlobNameInvalidError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_logical_blob_name_invalid_error();
  return error;
}

Error Error::OpNameExistError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_op_name_exist_error();
  return error;
}

Error Error::OpConfDeviceTagNoSetError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_op_conf_device_tag_no_set_error();
  return error;
}

Error Error::PlacementError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_placement_error();
  return error;
}

Error Error::BlobSplitAxisInferError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_blob_split_axis_infer_error();
  return error;
}

Error Error::UnknownJobBuildAndInferError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_unknown_job_build_and_infer_error();
  return error;
}

Error Error::CheckFailedError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_check_failed_error();
  return error;
}

Error Error::Todo() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_todo_error();
  return error;
}

Error Error::Unimplemented() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_unimplemented_error();
  return error;
}

Error Error::BoxingNotSupportedError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_boxing_not_supported_error();
  return error;
}

Error Error::OpKernelNotFoundError(const std::string& error_summary,
                                   const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->set_error_summary(error_summary);
  auto* op_kernel_not_found_error = error->mutable_op_kernel_not_found_error();
  for (const auto& msg : error_msgs) {
    op_kernel_not_found_error->add_op_kernels_not_found_debug_str(msg);
  }
  return error;
}

Error Error::MultipleOpKernelsMatchedError(const std::string& error_summary,
                                           const std::vector<std::string>& error_msgs) {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->set_error_summary(error_summary);
  auto* multiple_op_kernels_matched_error = error->mutable_multiple_op_kernels_matched_error();
  for (const auto& msg : error_msgs) {
    multiple_op_kernels_matched_error->add_matched_op_kernels_debug_str(msg);
  }
  return error;
}

Error Error::MemoryZoneOutOfMemoryError(int64_t machine_id, int64_t mem_zone_id, uint64_t calc,
                                        uint64_t available, const std::string& device_tag) {
  auto error = std::make_shared<cfg::ErrorProto>();
  auto* memory_zone_out_of_memory_error = error->mutable_memory_zone_out_of_memory_error();
  memory_zone_out_of_memory_error->add_machine_id(std::to_string(machine_id));
  memory_zone_out_of_memory_error->add_mem_zone_id(std::to_string(mem_zone_id));
  memory_zone_out_of_memory_error->add_device_tag(device_tag);
  memory_zone_out_of_memory_error->add_available(std::to_string(available) + " bytes");
  memory_zone_out_of_memory_error->add_required(std::to_string(calc) + " bytes");
  return error;
}

Error Error::LossBlobNotFoundError(const std::string& error_summary) {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_loss_blob_not_found_error();
  error->set_error_summary(error_summary);
  return error;
}

Error Error::RwMutexedObjectNotFoundError() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_rw_mutexed_object_not_found_error();
  return error;
}

Error Error::GradientFunctionNotFound() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_gradient_function_not_found_error();
  return error;
}

Error Error::SymbolIdUninitialized() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_symbol_id_uninitialized_error();
  return error;
}

Error Error::CompileOptionWrong() {
  auto error = std::make_shared<cfg::ErrorProto>();
  error->mutable_compile_option_wrong_error();
  return error;
}

void ThrowError(const std::shared_ptr<cfg::ErrorProto>& error) {
  *MutThreadLocalError() = error;
  CHECK_NE(error->error_type_case(), cfg::ErrorProto::ERROR_TYPE_NOT_SET);
  switch (error->error_type_case()) {
#define MAKE_ENTRY(cls)                                      \
  case cfg::ErrorProto::OF_PP_CAT(k, OF_PP_CAT(cls, Error)): \
    throw OF_PP_CAT(cls, Exception)(error->DebugString());

    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, EXCEPTION_SEQ)

#undef MAKE_ENTRY
    default: UNIMPLEMENTED();
  }
}

const std::shared_ptr<cfg::ErrorProto>& ThreadLocalError() { return *MutThreadLocalError(); }

}  // namespace oneflow
