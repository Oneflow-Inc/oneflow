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

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(const LogicalBlobId& lbi,
                                                            int32_t part_id) {
  // op_name_dir
  std::string op_name_dir = JoinPath(root_path_, lbi.op_name());
  OfCallOnce(op_name_dir, GlobalFS(), &fs::FileSystem::CreateDir);
  // bn_in_op_tmp_dir
  std::string bn_in_op_tmp_dir = JoinPath(op_name_dir, lbi.blob_name() + "_tmp4a58");
  LOG(INFO) << bn_in_op_tmp_dir;
  OfCallOnce(bn_in_op_tmp_dir, GlobalFS(), &fs::FileSystem::CreateDir);
  // part_file
  std::string part_file = JoinPath(bn_in_op_tmp_dir, "part_" + std::to_string(part_id));
  return std::make_unique<PersistentOutStream>(GlobalFS(), part_file);
}

void Snapshot::OnePartDone(const LogicalBlobId& lbi, int32_t part_id, int32_t part_num) {
  std::string lbn_done_key = JoinPath(root_path_, "part_done", lbi.op_name(), lbi.blob_name());
  int32_t lbn_done_cnt = Global<CtrlClient>::Get()->IncreaseCount(lbn_done_key);
  if (lbn_done_cnt == part_num) {
    Global<CtrlClient>::Get()->EraseCount(lbn_done_key);
    ConcatLbnFile(lbi, part_num, JoinPath(root_path_, lbi.op_name(), lbi.blob_name()));
  }
}

void Snapshot::ConcatLbnFile(const LogicalBlobId& lbi, int32_t part_num,
                             const std::string& concat_file) {
  std::vector<char> buffer(Global<JobDesc>::Get()->persistence_buf_byte());
  std::string part_dir = JoinPath(root_path_, lbi.op_name(), lbi.blob_name() + "_tmp4a58");
  {
    PersistentOutStream out_stream(GlobalFS(), concat_file);
    for (int32_t i = 0; i < part_num; ++i) {
      std::unique_ptr<fs::RandomAccessFile> part_file;
      std::string part_file_path = JoinPath(part_dir, "part_" + std::to_string(i));
      GlobalFS()->NewRandomAccessFile(part_file_path, &part_file);
      uint64_t part_file_size = GlobalFS()->GetFileSize(part_file_path);
      uint64_t offset = 0;
      while (offset < part_file_size) {
        uint64_t n = std::min(buffer.size(), part_file_size - offset);
        part_file->Read(offset, n, buffer.data());
        out_stream.Write(buffer.data(), n);
        offset += n;
      }
      GlobalFS()->DelFile(part_file_path);
    }
  }
  GlobalFS()->DeleteDir(part_dir);
  std::string snapshot_done_path = JoinPath(root_path_, "snapshot_done");
  int32_t snapshot_done_cnt = Global<CtrlClient>::Get()->IncreaseCount(snapshot_done_path);
  if (snapshot_done_cnt == Global<SnapshotMgr>::Get()->total_mbn_num()) {
    Global<CtrlClient>::Get()->EraseCount(snapshot_done_path);
    PersistentOutStream out_stream(GlobalFS(), snapshot_done_path);
  }
}

}  // namespace oneflow
