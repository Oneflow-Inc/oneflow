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
#ifndef ONEFLOW_CORE_VM_STREAM_TYPE_ID_H_
#define ONEFLOW_CORE_VM_STREAM_TYPE_ID_H_

#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/object_msg/flat_msg.h"

namespace oneflow {
namespace vm {

class StreamType;

class StreamTypeId final {
 public:
  using self_type = StreamTypeId;
  StreamTypeId() { __Init__(); }
  StreamTypeId(const StreamTypeId& rhs) { __Init__(rhs.stream_type_, rhs.interpret_type_); }
  void __Init__() { std::memset(this, 0, sizeof(StreamTypeId)); }
  void __Init__(const StreamType* stream_type, InterpretType interpret_type) {
    __Init__();
    stream_type_ = stream_type;
    interpret_type_ = interpret_type;
  }
  void __Init__(const StreamType* stream_type) { __Init__(stream_type, InterpretType::kCompute); }
  void __Delete__() { clear(); }
  void clear() {
    stream_type_ = nullptr;
    interpret_type_ = InterpretType::kInvalidInterpretType;
  }
  void CopyFrom(const StreamTypeId& rhs) { __Init__(rhs.stream_type_, rhs.interpret_type_); }

  const StreamType& stream_type() const { return *stream_type_; }
  InterpretType interpret_type() const { return interpret_type_; }

  void set_interpret_type(InterpretType val) { interpret_type_ = val; }

  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();

 private:
  const StreamType* stream_type_;
  InterpretType interpret_type_;
};

}  // namespace vm
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::vm::StreamTypeId> {
  size_t operator()(const oneflow::vm::StreamTypeId& stream_type_id) const {
    return std::hash<const oneflow::vm::StreamType*>()(&stream_type_id.stream_type())
           ^ std::hash<int>()(stream_type_id.interpret_type());
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_VM_STREAM_TYPE_ID_H_
