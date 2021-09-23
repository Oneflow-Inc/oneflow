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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_H_

#include <memory>
#include <string>
#include <unordered_set>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream_descriptor.h"

namespace oneflow {

class LocalDepObject;

class Stream final {
 public:
  explicit Stream(const Stream&) = default;
  explicit Stream(Stream&&) = default;
  ~Stream() = default;
  Stream& operator=(const Stream&) = delete;

  // new stream by default device.
  static Maybe<Symbol<Stream>> (*NewByDefaultDevice)(const std::string& stream_type_name);

  // new stream by default stream_type_name.
  static Maybe<Symbol<Stream>> (*NewByDefaultName)(Symbol<Device> device);

  static Maybe<Symbol<Stream>> (*New)(const std::string& stream_type_name, Symbol<Device> device);

  const StreamDescriptor& stream_descriptor() const { return *stream_descriptor_; }
  const Symbol<Device>& device() const { return device_; }

  bool operator==(const Stream& other) const {
    return this->stream_descriptor_ == other.stream_descriptor_ && this->device_ == other.device_;
  }

  size_t CalcHashValue() const {
    return std::hash<const StreamDescriptor*>()(stream_descriptor_)
           ^ std::hash<Symbol<Device>>()(device_);
  }

  static Maybe<Symbol<Stream>> RawNew(const StreamDescriptor* stream_descriptor,
                                      Symbol<Device> device);

  Optional<LocalDepObject*> mut_transport_local_dep_object() const {
    return transport_local_dep_object_;
  };
  LocalDepObject* mut_schedule_local_dep_object() const { return schedule_local_dep_object_; }

  size_t* mut_local_dep_object_pool_index() const { return &local_dep_object_pool_index_; }

 private:
  Stream(const StreamDescriptor* stream_descriptor, Symbol<Device> device)
      : stream_descriptor_(stream_descriptor), device_(device), local_dep_object_pool_index_(0) {}

  Maybe<void> Init();

  // hash key fields.
  const StreamDescriptor* stream_descriptor_;
  Symbol<Device> device_;

  mutable size_t local_dep_object_pool_index_;

  Optional<LocalDepObject*> transport_local_dep_object_;
  LocalDepObject* schedule_local_dep_object_;
};

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::Stream> final {
  size_t operator()(const oneflow::Stream& stream) const { return stream.CalcHashValue(); }
};
}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_H_
