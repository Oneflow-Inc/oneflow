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
#include "oneflow/core/common/stream_role.h"
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
    return this->device() == that.device() && this->stream_role() == that.stream_role();
  }
  bool operator!=(const Stream& that) const { return !(*this == that); }

  static Maybe<Symbol<Stream>> New(Symbol<Device> device, StreamRole stream_role);

  Symbol<Device> device() const { return device_; }
  StreamRole stream_role() const { return stream_role_; }
  size_t unique_stream_id() const { return unique_stream_id_; }

 private:
  Stream(Symbol<Device> device, StreamRole stream_role);

  static Maybe<Symbol<Stream>> RawNew(Symbol<Device> device, StreamRole stream_role);

  Maybe<void> Init(size_t unique_stream_id);

  Symbol<Device> device_;
  StreamRole stream_role_;
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
           ^ std::hash<StreamRole>()(stream.stream_role());
  }
};

}  // namespace std
#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_H_
