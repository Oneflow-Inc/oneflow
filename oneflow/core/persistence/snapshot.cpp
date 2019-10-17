#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace {

std::string GenDataFilePath(const std::string& root, const std::string& key) {
  return JoinPath(root, key);
}

}  // namespace

SnapshotReader::SnapshotReader(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path) {}

void SnapshotReader::Read(const std::string& key, Blob* blob) const {
  Read(key, blob->shape(), blob->data_type(), TensorSliceView(blob->shape()),
       blob->mut_dptr<char>());
}

void SnapshotReader::Read(const std::string& key, const Shape& logical_blob_shape,
                          DataType data_type, const TensorSliceView& slice, char* dst) const {
  const TensorSliceView logical_blob_slice(logical_blob_shape);
  CHECK(logical_blob_slice.Contains(slice));
  const std::string path = GenDataFilePath(root_path_, key);
  const int64_t logical_blob_size = logical_blob_shape.elem_cnt() * GetSizeOfDataType(data_type);
  CHECK_EQ(SnapshotFS()->GetFileSize(path), logical_blob_size)
      << "unexpected model snapshot size, path: " << path;
  if (slice.shape().Count(1) == logical_blob_shape.Count(1)) {
    PersistentInStream in_stream(
        SnapshotFS(), path,
        slice.At(0).begin() * slice.shape().Count(1) * GetSizeOfDataType(data_type));
    in_stream.Read(dst, slice.shape().elem_cnt() * GetSizeOfDataType(data_type));
  } else {
    std::vector<char> buffer(logical_blob_size);
    PersistentInStream in_stream(SnapshotFS(), path);
    in_stream.Read(buffer.data(), logical_blob_size);
    TensorSliceCopier copier(slice, logical_blob_slice, data_type);
    CpuDeviceCtx device_ctx;
    std::unique_ptr<MemoryCopier> host_memory_copier(NewDefaultMemoryCopier(DeviceType::kCPU));
    copier.Copy(&device_ctx, *host_memory_copier, dst, buffer.data());
  }
}

void SnapshotReader::Read(const std::string& key, const Shape& logical_blob_shape,
                          const TensorSliceView& slice, Blob* blob) const {
  CHECK_EQ(slice.shape(), blob->shape());
  Read(key, logical_blob_shape, blob->data_type(), slice, blob->mut_dptr<char>());
}

void SnapshotReader::Close() {}

SnapshotWriter::SnapshotWriter(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path) {
  OfCallOnce("SnapshotWriteCheckRootPath-" + snapshot_root_path, [&]() {
    if (SnapshotFS()->FileExists(snapshot_root_path)) {
      CHECK(SnapshotFS()->IsDirectory(snapshot_root_path))
          << "root directory of model snapshot not found, path: " << snapshot_root_path;
      CHECK(SnapshotFS()->IsDirEmpty(snapshot_root_path))
          << "root directory of model snapshot not empty, path: " << snapshot_root_path;
    } else {
      SnapshotFS()->CreateDir(snapshot_root_path);
    }
  });
}

void SnapshotWriter::Write(const std::string& key, const char* data, size_t size) {
  const std::string path = GenDataFilePath(root_path_, key);
  const std::string dir_path = Dirname(path);
  SnapshotFS()->CreateDirIfNotExist(dir_path);
  CHECK(!SnapshotFS()->FileExists(path));
  PersistentOutStream out_stream(SnapshotFS(), path);
  out_stream.Write(data, size);
}

void SnapshotWriter::Write(const std::string& key, const Blob* blob) {
  Write(key, blob->dptr<char>(), blob->ByteSizeOfDataContentField());
}

void SnapshotWriter::Close() {
  PersistentOutStream out_stream(SnapshotFS(), JoinPath(root_path_, "snapshot_done"));
}

}  // namespace oneflow
