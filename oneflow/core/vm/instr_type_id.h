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
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/common/layout_standardize.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/interpret_type.h"

namespace oneflow {
namespace vm {

class InstructionType;

class InstrTypeId final {
 public:
  InstrTypeId() : instruction_type_(nullptr) { __Init__(); }
  InstrTypeId(const InstrTypeId& rhs) : instruction_type_(nullptr) {
    __Init__();
    CopyFrom(rhs);
  }

  ~InstrTypeId() = default;

  void __Init__() {
    std::memset(this, 0, sizeof(InstrTypeId));
    mutable_stream_type_id()->__Init__();
  }
  void __Init__(const StreamType* stream_type, const InstructionType* instruction_type,
                InterpretType interpret_type) {
    __Init__();
    mutable_stream_type_id()->__Init__(stream_type, interpret_type);
    instruction_type_ = instruction_type;
  }
  void clear() {
    stream_type_id_.clear();
    instruction_type_ = nullptr;
  }
  void CopyFrom(const InstrTypeId& rhs) {
    stream_type_id_.CopyFrom(rhs.stream_type_id_);
    instruction_type_ = &rhs.instruction_type();
  }
  // Getters
  const StreamTypeId& stream_type_id() const { return stream_type_id_; }
  const InstructionType& instruction_type() const { return *instruction_type_; }

  // Setters
  StreamTypeId* mut_stream_type_id() { return &stream_type_id_; }
  StreamTypeId* mutable_stream_type_id() { return &stream_type_id_; }

  bool operator==(const InstrTypeId& rhs) const {
    return stream_type_id_ == rhs.stream_type_id_ && instruction_type_ == rhs.instruction_type_;
  }
  bool operator<(const InstrTypeId& rhs) const {
    if (!(stream_type_id_ == rhs.stream_type_id_)) { return stream_type_id_ < rhs.stream_type_id_; }
    if (!(instruction_type_ == rhs.instruction_type_)) {
      return instruction_type_ < rhs.instruction_type_;
    }
    return false;
  }
  bool operator<=(const InstrTypeId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  const InstructionType* instruction_type_;
  StreamTypeId stream_type_id_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
