#include "oneflow/core/common/str_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

Snapshot::Snapshot(const std::string& snapshot_root_path) : root_path_(snapshot_root_path) {
  CHECK(SnapshotFS()->IsDirectory(snapshot_root_path))
      << "root directory of model snapshot not found, path: " << snapshot_root_path;
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(const LogicalBlobId& lbi) {
  const std::string op_name_dir = JoinPath(root_path_, lbi.op_name());
  SnapshotFS()->CreateDir(op_name_dir);
  return std::make_unique<PersistentOutStream>(SnapshotFS(),
                                               JoinPath(op_name_dir, lbi.blob_name()));
}

void Snapshot::Done() {
  PersistentOutStream out_stream(SnapshotFS(), JoinPath(root_path_, "snapshot_done"));
}

namespace {

std::string GenDataFilePath(const std::string& root, const std::string& key) {
  return JoinPath(root, key);
}

std::string DirName(const std::string& path) {
  const size_t pos = path.rfind("\\/");
  if (pos == std::string::npos) {
    return "";
  } else {
    return path.substr(0, pos + 1);
  }
}

}  // namespace

SnapshotReader::SnapshotReader(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path) {}

void SnapshotReader::Read(const std::string& key, Blob* blob) const {
  const std::string path = GenDataFilePath(root_path_, key);
  const int64_t blob_size = blob->ByteSizeOfDataContentField();
  CHECK_EQ(SnapshotFS()->GetFileSize(path), blob_size);
  PersistentInStream in_stream(SnapshotFS(), path);
  in_stream.Read(blob->mut_dptr<char>(), blob_size);
}

void SnapshotReader::Close() {}

SnapshotWriter::SnapshotWriter(const std::string& snapshot_root_path)
    : root_path_(snapshot_root_path), closed_(false), writing_count_(0) {
  if (SnapshotFS()->FileExists(snapshot_root_path)) {
    CHECK(SnapshotFS()->IsDirectory(snapshot_root_path))
        << "root directory of model snapshot not found, path: " << snapshot_root_path;
    CHECK(SnapshotFS()->IsDirEmpty(snapshot_root_path));
  } else {
    SnapshotFS()->CreateDir(snapshot_root_path);
  }
}

void SnapshotWriter::Write(const std::string& key, const Blob* blob) {
  const std::string path = GenDataFilePath(root_path_, key);
  const std::string dir = DirName(path);
  {
    std::unique_lock<std::mutex> lck(writer_mutex_);
    CHECK(!closed_);
    writing_count_ += 1;
    SnapshotFS()->CreateDirIfNotExist(dir);
    CHECK(!SnapshotFS()->FileExists(path));
  }
  const int64_t blob_size = blob->ByteSizeOfDataContentField();
  PersistentOutStream out_stream(SnapshotFS(), path);
  out_stream.Write(blob->dptr<char>(), blob_size);
  {
    std::unique_lock<std::mutex> lck(writer_mutex_);
    writing_count_ -= 1;
  }
}

void SnapshotWriter::Close() {
  std::unique_lock<std::mutex> lck(writer_mutex_);
  CHECK_EQ(writing_count_, 0);
  closed_ = true;
  PersistentOutStream out_stream(SnapshotFS(), JoinPath(root_path_, "snapshot_done"));
}

}  // namespace oneflow
