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

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/registry_error.h"
#include "oneflow/core/framework/stream_descriptor.h"

namespace oneflow {

namespace {

// No need to lock because insert or registry actions are run before function main.
HashMap<std::string, const StreamDescriptor*>* Name2StreamDescriptor() {
  static HashMap<std::string, const StreamDescriptor*> map;
  return &map;
}

}  // namespace

Maybe<const StreamDescriptor*> StreamDescriptor::Find(const std::string& stream_type_name) {
  return JUST(MapAt(*Name2StreamDescriptor(), stream_type_name));
}

Maybe<void> StreamDescriptor::Register(
    const std::string& stream_type_name, const std::string& device_type,
    const Optional<std::string>& shared_transport_stream_type_name,
    const std::string& shared_schedule_stream_type_name,
    const Optional<std::string>& local_call_instruction_name, size_t local_dep_object_pool_size) {
  const auto* stream_descriptor = new StreamDescriptor(
      stream_type_name, device_type, shared_transport_stream_type_name,
      shared_schedule_stream_type_name, local_call_instruction_name, local_dep_object_pool_size);
  CHECK_OR_RETURN(Name2StreamDescriptor()->emplace(stream_type_name, stream_descriptor).second)
      << stream_type_name << " has been registered";
  return Maybe<void>::Ok();
}

bool StreamDescriptor::need_soft_sync_stream() const {
  // TODO(jianhao): refactor this code by instroducing vm::StreamType into StreamDescriptor
  return local_call_instruction_name().has_value()
         && *CHECK_JUST(local_call_instruction_name()) == "gpu.LocalCallOpKernel";
}

namespace {

template<typename... Args>
void RegisterStreamDescriptor(Args&&... args) {
  CatchRegistryError(
      [&]() -> Maybe<void> { return StreamDescriptor::Register(std::forward<Args>(args)...); });
}

}  // namespace

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"cpu",
    /*device_type=*/"cpu",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"cpu",
    /*local_call_instruction_name=*/"cpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"gpu",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"cuda",
    /*local_call_instruction_name=*/"gpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"cuda",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"cuda",
    /*local_call_instruction_name=*/"gpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"cuda_h2d",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"cuda_h2d",
    /*local_call_instruction_name=*/"cuda_h2d.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/kDoubleBufferPoolSize));

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"cuda_d2h",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"cuda_d2h",
    /*local_call_instruction_name=*/"cuda_d2h.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/kDoubleBufferPoolSize));

// share scheduling LocalDepObject between comm_net and sync_launched_nccl
COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"comm_net",
    /*device_type=*/"cpu",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"comm_net",
    /*local_call_instruction_name=*/"cpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

// share scheduling LocalDepObject between comm_net and sync_launched_nccl
// share transport LocalDepObject between sync_launched_nccl and async_launched_nccl
COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"sync_launched_nccl",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/"async_launched_nccl",
    /*shared_schedule_stream_type_name=*/"comm_net",
    /*local_call_instruction_name=*/"gpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

// share transport LocalDepObject between sync_launched_nccl and async_launched_nccl
COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"async_launched_nccl",
    /*device_type=*/"cuda",
    /*shared_transport_stream_type_name=*/"async_launched_nccl",
    /*shared_schedule_stream_type_name=*/"async_launched_nccl",
    /*local_call_instruction_name=*/"async.gpu.LocalCallOpKernel",
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

COMMAND(RegisterStreamDescriptor(
    /*stream_type_name=*/"critical_section",
    /*device_type=*/"cpu",
    /*shared_transport_stream_type_name=*/NullOpt,
    /*shared_schedule_stream_type_name=*/"critical_section",
    /*local_call_instruction_name=*/NullOpt,
    /*local_dep_object_pool_size=*/GetInstructionHighWaterMark()));

}  // namespace oneflow
