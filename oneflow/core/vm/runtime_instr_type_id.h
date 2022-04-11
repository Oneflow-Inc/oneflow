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
#ifndef ONEFLOW_CORE_VM_RUNTIME_INSTR_TYPE_ID_H_
#define ONEFLOW_CORE_VM_RUNTIME_INSTR_TYPE_ID_H_

#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/stream_runtime_desc.h"

namespace oneflow {
namespace vm {

class RtInstrTypeId final {
 public:
  RtInstrTypeId(const RtInstrTypeId&) = default;
  RtInstrTypeId(RtInstrTypeId&&) = default;
  ~RtInstrTypeId() = default;

  RtInstrTypeId(const InstrTypeId& instr_type_id, StreamRtDesc* stream_rt_desc)
      : instr_type_id_(instr_type_id), stream_rt_desc_(stream_rt_desc) {
    if (stream_rt_desc->stream_type().IsControlStreamType()) {
      get_stream_ = &StreamRtDesc::GetSoleStream;
    } else {
      get_stream_ = &StreamRtDesc::GetDeviceStream;
    }
  }

  const InstrTypeId& instr_type_id() const { return instr_type_id_; }
  Stream* GetStream(int device_id) const { return (stream_rt_desc_->*get_stream_)(device_id); }

 private:
  const InstrTypeId instr_type_id_;
  StreamRtDesc* stream_rt_desc_;
  Stream* (StreamRtDesc::*get_stream_)(int device_id) const;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_RUNTIME_INSTR_TYPE_ID_H_
