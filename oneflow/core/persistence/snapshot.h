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
