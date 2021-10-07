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
#ifndef ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC_MSG_H_
#define ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC_MSG_H_

#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {
namespace vm {

class StreamType;
struct StreamDesc;

// Rt is short for Runtime
// clang-format off
OBJECT_MSG_BEGIN(StreamRtDesc);
 public:
  StreamRtDesc() = default;
  // types
  using StreamId2Stream = intrusive::SkipList<OBJECT_MSG_FIELD(Stream, stream_id_)>;
  // Getters
  const StreamDesc& stream_desc() const {
    if (stream_desc_) { return stream_desc_.Get(); }
    static const auto default_val = ObjectMsgPtr<StreamDesc>::New();
    return default_val.Get();
  }
  const StreamTypeId& stream_type_id() const { return stream_type_id_.key().Get(); }
  const StreamId2Stream& stream_id2stream() const { return stream_id2stream_; }
  // Setters
  StreamDesc* mut_stream_desc() { return mutable_stream_desc(); }
  StreamDesc* mutable_stream_desc() { 
    if (!stream_desc_) { stream_desc_ = ObjectMsgPtr<StreamDesc>::New(); }
    return stream_desc_.Mutable();
  }
  void reset_stream_desc(StreamDesc* stream_desc) { stream_desc_.Reset(stream_desc); }
  StreamTypeId* mut_stream_type_id() { return stream_type_id_.mut_key()->Mutable(); }
  StreamTypeId* mutable_stream_type_id() { return stream_type_id_.mut_key()->Mutable(); }
  StreamId2Stream* mut_stream_id2stream() { return &stream_id2stream_; }
  StreamId2Stream* mutable_stream_id2stream() { return &stream_id2stream_; }

  // methods
  OF_PUBLIC void __Init__(StreamDesc* stream_desc);
  OF_PUBLIC const StreamType& stream_type() const;

  // fields
  OBJECT_MSG_DEFINE_FIELD(ObjectMsgPtr<StreamDesc>, stream_desc_); 

  // list entries
  using StreamTypeIdKey = intrusive::SkipListEntry<FlatMsg<StreamTypeId>, 7>;
  OBJECT_MSG_DEFINE_FIELD(StreamTypeIdKey, stream_type_id_);
  OBJECT_MSG_DEFINE_FIELD(StreamId2Stream, stream_id2stream_);
OBJECT_MSG_END(StreamRtDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC_MSG_H_
