#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

namespace {

std::string GenDataFilePath(const std::string& root, const std::string& key) {
  return JoinPath(root, key);
}

}  // namespace

SnapshotReader::SnapshotReader(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path) {}

void SnapshotReader::Read(const std::string& key, Blob* blob) const {
  const std::string path = GenDataFilePath(root_path_, key);
  const int64_t blob_size = blob->ByteSizeOfDataContentField();
  CHECK_EQ(SnapshotFS()->GetFileSize(path), blob_size)
      << "unexpected model snapshot size, path: " << path;
  PersistentInStream in_stream(SnapshotFS(), path);
  in_stream.ReadFully(blob->mut_dptr<char>(), blob_size);
}

void SnapshotReader::Close() {}

SnapshotWriter::SnapshotWriter(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path) {
  if (SnapshotFS()->FileExists(snapshot_root_path)) {
    CHECK(SnapshotFS()->IsDirectory(snapshot_root_path))
        << "root directory of model snapshot not found, path: " << snapshot_root_path;
    CHECK(SnapshotFS()->IsDirEmpty(snapshot_root_path))
        << "root directory of model snapshot not empty, path: " << snapshot_root_path;
  } else {
    SnapshotFS()->CreateDir(snapshot_root_path);
  }
}

void SnapshotWriter::Write(const std::string& key, const Blob* blob) {
  const std::string path = GenDataFilePath(root_path_, key);
  const std::string dir_path = Dirname(path);
  SnapshotFS()->CreateDirIfNotExist(dir_path);
  CHECK(!SnapshotFS()->FileExists(path));
  const int64_t blob_size = blob->ByteSizeOfDataContentField();
  PersistentOutStream out_stream(SnapshotFS(), path);
  out_stream.Write(blob->dptr<char>(), blob_size);
}

void SnapshotWriter::Close() {
  PersistentOutStream out_stream(SnapshotFS(), JoinPath(root_path_, "snapshot_done"));
}

}  // namespace oneflow
