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

#include <functional>
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

class Stream final {
 public:
  Stream(const Stream&) = default;
  Stream(Stream&&) = default;
  ~Stream() = default;

  bool operator==(const Stream& that) const {
    return this->device() == that.device() && this->stream_type() == that.stream_type();
  }
  bool operator!=(const Stream& that) const { return !(*this == that); }

  static Maybe<Symbol<Stream>> New(Symbol<Device> device, StreamType stream_type);

  Symbol<Device> device() const { return device_; }
  StreamType stream_type() const { return stream_type_; }
  size_t unique_stream_id() const { return unique_stream_id_; }

 private:
  Stream(Symbol<Device> device, StreamType stream_type);

  static Maybe<Symbol<Stream>> RawNew(Symbol<Device> device, StreamType stream_type);

  Maybe<void> Init(size_t unique_stream_id);

  Symbol<Device> device_;
  StreamType stream_type_;
  size_t unique_stream_id_;
};

extern Maybe<Symbol<Stream>> (*GetDefaultStreamByDevice)(Symbol<Device>);
class ParallelDesc;
extern Maybe<Symbol<Stream>> (*GetDefaultStreamByPlacement)(Symbol<ParallelDesc>);

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::Stream> final {
  size_t operator()(const oneflow::Stream& stream) const {
    using namespace oneflow;
    return std::hash<Symbol<Device>>()(stream.device())
           ^ std::hash<StreamType>()(stream.stream_type());
  }
};

}  // namespace std
#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_H_
