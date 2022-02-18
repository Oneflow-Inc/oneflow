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
#ifndef ONEFLOW_CORE_VM_VPU_DESC__H_
#define ONEFLOW_CORE_VM_VPU_DESC__H_

#include <cstring>
#include <typeindex>
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/id_util.h"

namespace oneflow {
namespace vm {

class StreamType;

class StreamId final {
 public:
  using self_type = StreamId;
  void __Init__() {}
  void __Init__(const StreamType* stream_type, int64_t global_device_id) {
    stream_type_ = stream_type;
    global_device_id_ = global_device_id;
  }

  void CopyFrom(const StreamId& rhs) { __Init__(rhs.stream_type_, rhs.global_device_id_); }

  const StreamType& stream_type() const { return *stream_type_; }
  int64_t global_device_id() const { return global_device_id_; }

  bool operator==(const StreamId& rhs) const {
    return stream_type_ == rhs.stream_type_ && global_device_id_ == rhs.global_device_id_;
  }

  bool operator<(const StreamId& rhs) const {
    if (!(stream_type_ == rhs.stream_type_)) { return stream_type_ < rhs.stream_type_; }
    return global_device_id_ < rhs.global_device_id_;
  }
  bool operator<=(const StreamId& rhs) const { return *this < rhs || *this == rhs; }

 private:
  const StreamType* stream_type_;
  int64_t global_device_id_;
};

class StreamDesc final : public intrusive::Base {
 public:
  // Getters
  int32_t num_streams_per_machine() const { return num_streams_per_machine_; }
  int32_t num_streams_per_thread() const { return num_streams_per_thread_; }
  const StreamType& stream_type() const { return *stream_type_key_.key(); }
  // Setters
  void set_num_streams_per_machine(int32_t val) { num_streams_per_machine_ = val; }
  void set_num_streams_per_thread(int32_t val) { num_streams_per_thread_ = val; }
  void set_stream_type(const StreamType* stream_type) { *stream_type_key_.mut_key() = stream_type; }

  // methods
  void __Init__() {}
  void __Init__(const StreamType* stream_type, int32_t num_streams_per_machine,
                int32_t num_streams_per_thread);
  int32_t num_threads() const;
  int32_t parallel_num() const { return num_streams_per_machine(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  StreamDesc()
      : intrusive_ref_(),
        num_streams_per_machine_(),
        num_streams_per_thread_(),
        stream_type_key_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  int32_t num_streams_per_machine_;
  int32_t num_streams_per_thread_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<const StreamType*, 7> stream_type_key_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VPU_DESC__H_
