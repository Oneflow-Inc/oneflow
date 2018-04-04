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
  OF_CALL_ONCE(op_name_dir, GlobalFS()->CreateDir(op_name_dir));
  // bn_in_op_tmp_dir
  std::string bn_in_op_tmp_dir = JoinPath(op_name_dir, bn_in_op + "_tmp");
  OF_CALL_ONCE(bn_in_op_tmp_dir, GlobalFS()->CreateDir(bn_in_op_tmp_dir));
  // part_file
  std::string part_file =
      JoinPath(bn_in_op_tmp_dir, "part_" + std::to_string(part_id));
  return of_make_unique<PersistentOutStream>(GlobalFS(), part_file);
}

void Snapshot::OnePartDone(const std::string& lbn, int32_t part_id,
                           int32_t part_num) {
  std::string lbn_done_key = JoinPath(root_path_, "part_done", lbn);
  int32_t lbn_done_cnt = Global<CtrlClient>::Get()->IncreaseCount(lbn_done_key);
  if (lbn_done_cnt == part_num) {
    Global<CtrlClient>::Get()->EraseCount(lbn_done_key);
    ConcatLbnFile(lbn, part_num, JoinPath(root_path_, lbn));
  }
}

void Snapshot::ConcatLbnFile(const std::string& lbn, int32_t part_num,
                             const std::string& concat_file) {
  std::vector<char> buffer(
      Global<JobDesc>::Get()->persistence_buffer_byte_size());
  std::string part_dir = JoinPath(root_path_, lbn + "_tmp");
  {
    PersistentOutStream out_stream(GlobalFS(), concat_file);
    for (int32_t i = 0; i < part_num; ++i) {
      std::unique_ptr<fs::RandomAccessFile> part_file;
      std::string part_file_path =
          JoinPath(part_dir, "part_" + std::to_string(i));
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
  int32_t snapshot_done_cnt =
      Global<CtrlClient>::Get()->IncreaseCount(snapshot_done_path);
  if (snapshot_done_cnt == Global<SnapshotMgr>::Get()->total_mbn_num()) {
    Global<CtrlClient>::Get()->EraseCount(snapshot_done_path);
    PersistentOutStream out_stream(GlobalFS(), snapshot_done_path);
  }
}

}  // namespace oneflow
