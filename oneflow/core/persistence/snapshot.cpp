#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

Snapshot::Snapshot(const std::string& snapshot_root_path) {
  CHECK(GlobalFS()->IsDirectory(snapshot_root_path));
  root_path_ = snapshot_root_path;
}

std::string Snapshot::GetDirFromOpName(const std::string& op_name) const {
  return JoinPath(root_path_, op_name);
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(
    const std::string& lbn, int32_t part_id) {
  // parse lbn
  std::pair<std::string, std::string> parsed_lbn = ParseLbn(lbn);
  const std::string& op_name = parsed_lbn.first;
  const std::string& bn_in_op = parsed_lbn.second;
  // op_name_dir
  std::string op_name_dir = JoinPath(root_path_, op_name);
  OF_ONCE_GUARD(op_name_dir, GlobalFS()->CreateDir(op_name_dir));
  // bn_in_op_tmp_dir
  std::string bn_in_op_tmp_dir = JoinPath(op_name_dir, bn_in_op + "_tmp");
  OF_ONCE_GUARD(bn_in_op_tmp_dir, GlobalFS()->CreateDir(bn_in_op_tmp_dir));
  // part_file
  std::string part_file =
      JoinPath(bn_in_op_tmp_dir, "part_" + std::to_string(part_id));
  return of_make_unique<PersistentOutStream>(GlobalFS(), part_file);
}

void Snapshot::OnePartDone(const std::string& lbn, int32_t part_id,
                           int32_t part_num) {
  std::string done_dir = JoinPath(root_path_, lbn + "_done");
  OF_ONCE_GUARD(done_dir, GlobalFS()->CreateDir(done_dir));
  std::string done_file_path = JoinPath(done_dir, std::to_string(part_id));
  CHECK_EQ(GlobalFS()->FileExists(done_file_path), false);
  { PersistentOutStream out_stream(GlobalFS(), done_file_path); }
  if (GlobalFS()->GetChildrenNumOfDir(done_dir) == part_num) {
    std::string concat_file = JoinPath(root_path_, lbn);
    OF_ONCE_GUARD(concat_file, GlobalFS()->DeleteRecursively(done_dir);
                  ConcatLbnFile(lbn, part_num, concat_file));
  }
}

void Snapshot::ConcatLbnFile(const std::string& lbn, int32_t part_num,
                             const std::string& concat_file) {
  static const uint64_t buffer_size = 256 * 1024 * 1024;
  static char* buffer = new char[buffer_size];
  std::pair<std::string, std::string> parsed_lbn = ParseLbn(lbn);
  const std::string& op_name = parsed_lbn.first;
  const std::string& bn_in_op = parsed_lbn.second;
  std::string part_dir = JoinPath(root_path_, lbn + "_tmp");
  {
    PersistentOutStream out_stream(GlobalFS(), concat_file);
    for (int32_t i = 0; i < part_num; ++i) {
      std::unique_ptr<fs::RandomAccessFile> part_file;
      std::string part_file_path =
          JoinPath(part_dir, "part_" + std::to_string(i));
      GlobalFS()->NewRandomAccessFile(part_file_path, &part_file);
      uint64_t part_file_size = 0;
      GlobalFS()->GetFileSize(part_file_path, &part_file_size);
      uint64_t offset = 0;
      while (offset < part_file_size) {
        uint64_t n = std::min(buffer_size, part_file_size - offset);
        part_file->Read(offset, n, buffer);
        out_stream.Write(buffer, n);
        offset += n;
      }
      GlobalFS()->DeleteFile(part_file_path);
    }
  }
  GlobalFS()->DeleteDir(part_dir);
  std::string done_dir = JoinPath(root_path_, "snapshot_done_tmp");
  OF_ONCE_GUARD(done_dir, GlobalFS()->CreateDir(done_dir));
  {
    PersistentOutStream out_stream(
        GlobalFS(), JoinPath(done_dir, op_name + "_" + bn_in_op));
  }
  if (GlobalFS()->GetChildrenNumOfDir(done_dir)
      == SnapshotMgr::Singleton()->num_of_model_blobs()) {
    std::string done_file = JoinPath(root_path_, "snapshot_done");
    OF_ONCE_GUARD(done_file, GlobalFS()->DeleteRecursively(done_dir);
                  { PersistentOutStream out_stream(GlobalFS(), done_file); });
  }
}

}  // namespace oneflow
