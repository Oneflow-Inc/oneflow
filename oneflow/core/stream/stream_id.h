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
#ifndef ONEFLOW_CORE_STREAM_STREAM_ID_H_
#define ONEFLOW_CORE_STREAM_STREAM_ID_H_

#include "oneflow/core/device/device_id.h"

namespace oneflow {

class StreamId {
 public:
  using index_t = uint32_t;

  constexpr static size_t kStreamIndexBits = 12;
  constexpr static index_t kMaxStreamIndex = (index_t{1} << kStreamIndexBits) - index_t{1};

  StreamId(const DeviceId& device_id, index_t stream_index)
      : device_id_(device_id), stream_index_(stream_index) {
    CHECK_LE(stream_index, kMaxStreamIndex);
  }
  StreamId(DeviceId::index_t node_index, DeviceType device_type, DeviceId::index_t device_index,
           index_t stream_index)
      : device_id_(node_index, device_type, device_index), stream_index_(stream_index) {
    CHECK_LE(stream_index, kMaxStreamIndex);
  }

  const DeviceId& device_id() const { return device_id_; }
  DeviceId::index_t node_index() const { return device_id_.node_index(); }
  DeviceType device_type() const { return device_id_.device_type(); }
  DeviceId::index_t device_index() const { return device_id_.device_index(); }
  index_t stream_index() const { return stream_index_; }

  bool operator==(const StreamId& rhs) const {
    return device_id_ == rhs.device_id_ && stream_index_ == rhs.stream_index_;
  }

  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }

  size_t hash() const {
    size_t hash = device_id_.hash();
    HashCombine(&hash, std::hash<index_t>{}(stream_index_));
    return hash;
  }

 private:
  DeviceId device_id_;
  index_t stream_index_;
};

int64_t EncodeStreamIdToInt64(const StreamId&);
StreamId DecodeStreamIdFromInt64(int64_t);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::StreamId> {
  size_t operator()(const oneflow::StreamId& stream_id) const { return stream_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_STREAM_STREAM_ID_H_
