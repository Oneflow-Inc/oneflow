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
#ifndef ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC__H_
#define ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC__H_

#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {
namespace vm {

class StreamType;
struct StreamDesc;

// Rt is short for Runtime
// clang-format off
INTRUSIVE_BEGIN(StreamRtDesc);
 public:
  // types
  using StreamId2Stream = intrusive::SkipList<INTRUSIVE_FIELD(Stream, stream_id_)>;
  // Getters
  const StreamDesc& stream_desc() const {
    if (stream_desc_) { return stream_desc_.Get(); }
    static const auto default_val = intrusive::make_shared<StreamDesc>();
    return default_val.Get();
  }
  const StreamTypeId& stream_type_id() const { return stream_type_id_.key().Get(); }
  const StreamId2Stream& stream_id2stream() const { return stream_id2stream_; }
  // Setters
  StreamDesc* mut_stream_desc() { 
    if (!stream_desc_) { stream_desc_ = intrusive::make_shared<StreamDesc>(); }
    return stream_desc_.Mutable();
  }
  void reset_stream_desc(StreamDesc* stream_desc) { stream_desc_.Reset(stream_desc); }
  StreamTypeId* mut_stream_type_id() { return stream_type_id_.mut_key()->Mutable(); }
  StreamId2Stream* mut_stream_id2stream() { return &stream_id2stream_; }

  // methods
  void __Init__(StreamDesc* stream_desc);
  const StreamType& stream_type() const;

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  StreamRtDesc() : intrusive_ref_(), stream_desc_(), stream_type_id_(), stream_id2stream_() {}
  INTRUSIVE_DEFINE_FIELD(intrusive::Ref, intrusive_ref_);
  // fields
  INTRUSIVE_DEFINE_FIELD(intrusive::shared_ptr<StreamDesc>, stream_desc_); 
  // list hooks
  using StreamTypeIdKey = intrusive::SkipListHook<FlatMsg<StreamTypeId>, 7>;
  INTRUSIVE_DEFINE_FIELD(StreamTypeIdKey, stream_type_id_);
  INTRUSIVE_DEFINE_FIELD(StreamId2Stream, stream_id2stream_);
INTRUSIVE_END(StreamRtDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC__H_
