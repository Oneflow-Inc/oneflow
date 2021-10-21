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
#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

class Blob;

class SnapshotReader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotReader);
  SnapshotReader() = delete;
  explicit SnapshotReader(const std::string& snapshot_root_path);
  ~SnapshotReader() = default;

  void Read(const std::string& key, const Shape& logical_blob_shape, DataType data_type,
            const TensorSliceView& slice, char* dst) const;
  void Read(const std::string& key, const Shape& logical_blob_shape, const TensorSliceView& slice,
            Blob* blob) const;
  void Read(const std::string& key, Blob* blob) const;
  bool HasKey(const std::string& key) const;
  void Close();

 private:
  const std::string root_path_;
};

class SnapshotWriter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotWriter);
  SnapshotWriter() = delete;
  explicit SnapshotWriter(const std::string& snapshot_root_path);
  ~SnapshotWriter() = default;

  void Write(const std::string& key, const char* data, size_t size);
  void Write(const std::string& key, const Blob* blob);
  void Close();

 private:
  const std::string root_path_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
