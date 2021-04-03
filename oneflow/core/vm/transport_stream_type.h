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
#ifndef ONEFLOW_CORE_VM_TRANSPORT_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_TRANSPORT_STREAM_TYPE_H_

#include <atomic>
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class TransportStreamType : public StreamType {
 public:
  TransportStreamType() = default;
  virtual ~TransportStreamType() = default;

  using RefCntType = std::atomic<int32_t>;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override;

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Compute(Instruction* instruction) const override;

  template<typename DerivedT>
  ObjectMsgPtr<StreamDesc> MakeTransportStreamDesc(const Resource& resource,
                                                   int64_t this_machine_id) const;
  bool SharingVirtualMachineThread() const override { return false; }
};

class TransportSenderStreamType : public TransportStreamType {
 public:
  TransportSenderStreamType() = default;
  ~TransportSenderStreamType() override = default;

  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
};

class TransportReceiverStreamType : public TransportStreamType {
 public:
  TransportReceiverStreamType() = default;
  ~TransportReceiverStreamType() override = default;

  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TRANSPORT_STREAM_TYPE_H_
