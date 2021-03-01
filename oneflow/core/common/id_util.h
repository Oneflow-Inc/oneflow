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
#ifndef ONEFLOW_CORE_COMMON_ID_UTIL_H_
#define ONEFLOW_CORE_COMMON_ID_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

// TaskId encode (may be extended to 128 bit in future)
// |            rank            | device_type | device_index  |                           |
// | ----------- 19 ----------- | ---- 5 ---- | ----- 7 ----- |                           |
// |                        DeviceId                          | stream_index |            |
// | ------------------------- 31 --------------------------- | ---- 12 ---- |            |
// |                               StreamId                                  | task_index |
// | -------------------------------- 43 ----------------------------------- | --- 21 --- |
// |                                      TaskId                                          |
// | ----------------------------------- 64 bit ----------------------------------------- |

class DeviceId {
 public:
  using rank_t = uint32_t;
  using device_index_t = uint32_t;

  constexpr static size_t kRankBits = 19;
  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static rank_t kMaxRank = (rank_t{1} << kRankBits) - rank_t{1};
  constexpr static size_t kMaxDeviceTypeVal = (size_t{1} << kDeviceTypeBits) - size_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};
  constexpr static device_index_t kCPUDeviceIndex = 0;

  DeviceId(rank_t rank, DeviceType device_type, device_index_t device_index)
      : rank_(rank), device_type_(device_type), device_index_(device_index) {
    CHECK_LE(rank, kMaxRank);
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }
  rank_t rank() const { return rank_; }
  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  bool operator==(const DeviceId& rhs) const {
    return rank_ == rhs.rank_ && device_type_ == rhs.device_type_
           && device_index_ == rhs.device_index_;
  }
  bool operator!=(const DeviceId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = std::hash<rank_t>{}(rank_);
    HashCombine(&hash, std::hash<size_t>{}(static_cast<size_t>(device_type_)));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  rank_t rank_;
  DeviceType device_type_;
  device_index_t device_index_;
};

class StreamId {
 public:
  using stream_index_t = uint32_t;

  constexpr static size_t kStreamIndexBits = 12;
  constexpr static stream_index_t kMaxStreamIndex =
      (stream_index_t{1} << kStreamIndexBits) - stream_index_t{1};

  StreamId(const DeviceId& device_id, stream_index_t stream_index)
      : device_id_(device_id), stream_index_(stream_index) {
    CHECK_LE(stream_index, kMaxStreamIndex);
  }
  const DeviceId& device_id() const { return device_id_; }
  stream_index_t stream_index() const { return stream_index_; }
  bool operator==(const StreamId& rhs) const {
    return device_id_ == rhs.device_id_ && stream_index_ == rhs.stream_index_;
  }
  bool operator!=(const StreamId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = device_id_.hash();
    HashCombine(&hash, std::hash<stream_index_t>{}(stream_index_));
    return hash;
  }

 private:
  DeviceId device_id_;
  stream_index_t stream_index_;
};

class TaskId {
 public:
  using task_index_t = uint32_t;

  const static size_t kTaskIndexBits = 21;
  constexpr static task_index_t kMaxTaskIndex =
      (task_index_t{1} << kTaskIndexBits) - task_index_t{1};

  TaskId(const StreamId& stream_id, task_index_t task_index)
      : stream_id_(stream_id), task_index_(task_index) {
    CHECK_LE(task_index_, kMaxTaskIndex);
  }
  const StreamId& stream_id() const { return stream_id_; }
  task_index_t task_index() const { return task_index_; }
  bool operator==(const TaskId& rhs) const {
    return stream_id_ == rhs.stream_id_ && task_index_ == rhs.task_index_;
  }
  bool operator!=(const TaskId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = stream_id_.hash();
    HashCombine(&hash, std::hash<task_index_t>{}(task_index_));
    return hash;
  }

 private:
  StreamId stream_id_;
  task_index_t task_index_;
};

class MemZoneId {
 public:
  constexpr static size_t kDeviceTypeBits = 5;
  constexpr static size_t kDeviceIndexBits = 7;
  constexpr static size_t kMaxDeviceTypeVal = (size_t{1} << kDeviceTypeBits) - size_t{1};
  constexpr static device_index_t kMaxDeviceIndex =
      (device_index_t{1} << kDeviceIndexBits) - device_index_t{1};

  MemZoneId() {
    device_type_ = DeviceType::kCPU;
    device_index_ = 0;
  }
  MemZoneId(DeviceType device_type, device_index_t device_index)
      : device_type_(device_type), device_index_(device_index) {
    CHECK_LE(static_cast<size_t>(device_type), kMaxDeviceTypeVal);
    CHECK_LE(device_index, kMaxDeviceIndex);
  }
  DeviceType device_type() const { return device_type_; }
  device_index_t device_index() const { return device_index_; }
  bool operator==(const MemZoneId& rhs) const {
    return device_type_ == rhs.device_type_ && device_index_ == rhs.device_index_;
  }
  bool operator!=(const MemZoneId& rhs) const { return !(*this == rhs); }
  size_t hash() const {
    size_t hash = std::hash<size_t>{}(static_cast<size_t>(device_type_));
    HashCombine(&hash, std::hash<device_index_t>{}(device_index_));
    return hash;
  }

 private:
  DeviceType device_type_;
  device_index_t device_index_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DeviceId> {
  size_t operator()(const oneflow::DeviceId& device_id) const { return device_id.hash(); }
};

template<>
struct hash<oneflow::StreamId> {
  size_t operator()(const oneflow::StreamId& stream_id) const { return stream_id.hash(); }
};

template<>
struct hash<oneflow::TaskId> {
  size_t operator()(const oneflow::TaskId& task_id) const { return task_id.hash(); }
};

template<>
struct hash<oneflow::MemZoneId> {
  size_t operator()(const oneflow::MemZoneId& mem_zone_id) const { return mem_zone_id.hash(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_ID_UTIL_H_
