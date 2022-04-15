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
class StreamDesc;

// Rt is short for Runtime
class StreamRtDesc final : public intrusive::Base {
 public:
  // Getters
  const StreamDesc& stream_desc() const {
    if (stream_desc_) { return stream_desc_.Get(); }
    static const auto default_val = intrusive::make_shared<StreamDesc>();
    return default_val.Get();
  }
  const StreamType& stream_type() const { return *stream_type_key_.key(); }
  const std::vector<intrusive::shared_ptr<Stream>>& device_id2stream() const {
    return device_id2stream_;
  }

  // The value of `device_id` is ignored.
  Stream* GetSoleStream(int device_id) const { return GetSoleStream(); }
  Stream* GetSoleStream() const {
    CHECK_EQ(device_id2stream().size(), 1);
    return device_id2stream().at(0).get();
  }

  Stream* GetDeviceStream(int device_id) const { return device_id2stream().at(device_id).get(); }

  // Setters
  StreamDesc* mut_stream_desc() {
    if (!stream_desc_) { stream_desc_ = intrusive::make_shared<StreamDesc>(); }
    return stream_desc_.Mutable();
  }
  void reset_stream_desc(StreamDesc* stream_desc) { stream_desc_.Reset(stream_desc); }
  void set_stream_type(const StreamType* stream_type) { *stream_type_key_.mut_key() = stream_type; }
  void add_stream(intrusive::shared_ptr<Stream> stream) {
    CHECK_EQ(stream->device_id(), device_id2stream_.size());
    device_id2stream_.emplace_back(stream);
  }

  // methods
  void __Init__(StreamDesc* stream_desc);

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  StreamRtDesc() : intrusive_ref_(), stream_desc_(), device_id2stream_(), stream_type_key_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  intrusive::shared_ptr<StreamDesc> stream_desc_;
  // containers
  std::vector<intrusive::shared_ptr<Stream>> device_id2stream_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<const StreamType*, 7> stream_type_key_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_RUNTIME_DESC__H_
