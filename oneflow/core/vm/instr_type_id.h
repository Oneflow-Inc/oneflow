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
#ifndef ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_ID_H_

#include <typeindex>
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/common/layout_standardize.h"
#include "oneflow/core/vm/stream_desc.h"

namespace oneflow {
namespace vm {

class InstructionType;
class StreamType;

class InstrTypeId final {
 public:
  InstrTypeId() { __Init__(); }
  InstrTypeId(const InstrTypeId& rhs) {
    __Init__();
    CopyFrom(rhs);
  }

  ~InstrTypeId() = default;

  void __Init__() { clear(); }
  void __Init__(const StreamType* stream_type, const InstructionType* instruction_type) {
    __Init__();
    set_stream_type(stream_type);
    instruction_type_ = instruction_type;
  }
  void clear() {
    stream_type_ = nullptr;
    instruction_type_ = nullptr;
  }
  void CopyFrom(const InstrTypeId& rhs) {
    stream_type_ = &rhs.stream_type();
    instruction_type_ = &rhs.instruction_type();
  }
  // Getters
  const StreamType& stream_type() const { return *stream_type_; }
  const InstructionType& instruction_type() const { return *instruction_type_; }

  // Setters
  void set_stream_type(const StreamType* stream_type) { stream_type_ = stream_type; }

  bool operator==(const InstrTypeId& rhs) const {
    return stream_type_ == rhs.stream_type_ && instruction_type_ == rhs.instruction_type_;
  }
  bool operator<(const InstrTypeId& rhs) const {
    if (!(stream_type_ == rhs.stream_type_)) { return stream_type_ < rhs.stream_type_; }
    if (!(instruction_type_ == rhs.instruction_type_)) {
      return instruction_type_ < rhs.instruction_type_;
    }
    return false;
  }
  bool operator<=(const InstrTypeId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  const InstructionType* instruction_type_;
  const StreamType* stream_type_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
