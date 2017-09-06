#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

namespace {

std::string MakeValidFileName(const std::string& key) {
  std::string valid_file_name;
  valid_file_name.reserve(key.size());
  for (char ch : key) {
    if (ch == '/') {
      valid_file_name.push_back('_');
    } else {
      valid_file_name.push_back(ch);
    }
  }
  return valid_file_name;
}

}  // namespace

const char* Snapshot::concat_file_name_ = "all";
const char* Snapshot::key_info_dir_name_ = "key_info";

Snapshot::Snapshot(const std::string& snapshot_root_path) {
  file_system_ = fs::GetFileSystem();
  FS_CHECK_OK(file_system_->IsDirectory(snapshot_root_path));
  root_path_ = snapshot_root_path;
  CheckAndConcat();
}

void Snapshot::CheckAndConcat() {
  // the children of the root path must be dir, not file
  std::vector<std::string> sub_dir_names;
  FS_CHECK_OK(file_system_->GetChildren(root_path_, &sub_dir_names));
  for (std::string sub_dir_name : sub_dir_names) {
    std::string sub_dir = JoinPath(root_path_, sub_dir_name);
    FS_CHECK_OK(file_system_->IsDirectory(sub_dir));
    // for the children of the sub_dir
    std::vector<std::string> file_names;
    FS_CHECK_OK(file_system_->GetChildren(sub_dir, &file_names));
    CHECK_NE(file_names.size(), 0);
    std::string concat_file_path = JoinPath(sub_dir, concat_file_name_);
    // for condition after concat
    // if the children number is 1 , the child must be the concated file named
    // "all"
    if (file_names.size() == 1) {
      std::string file_path = JoinPath(sub_dir, file_names[0]);
      FS_CHECK_OK(file_system_->FileExists(file_path));
      CHECK_EQ(file_names[0], concat_file_name_);
      continue;
    }
    // for condition before concat
    // if the children number is more than 1, the child must contain:
    //   1. a dir named "key_info"
    //   2. files named {0, 1, 2, 3,...,n-1} , n is the part num
    // and then CONCAT the files to one file, delete origin files and Dir
    //
    // first: check key_info
    std::string key_info_dir_path = JoinPath(sub_dir, key_info_dir_name_);
    FS_CHECK_OK(file_system_->IsDirectory(key_info_dir_path));
    std::vector<std::string> key_info_subs;
    FS_CHECK_OK(file_system_->GetChildren(key_info_dir_path, &key_info_subs));
    int32_t part_num = -1;
    for (std::string sub_file_name : key_info_subs) {
      if (sub_file_name.length() > 6 && sub_file_name.substr(0, 5) == "total") {
        part_num = oneflow_cast<int32_t>(sub_file_name.substr(6));
        break;
      }
    }
    CHECK_GT(part_num, 0);
    CHECK_EQ(part_num, key_info_subs.size() - 1);
    CHECK_EQ(part_num, file_names.size() - 1);
    for (size_t i = 0; i < part_num; ++i) {
      std::string done_file_path =
          JoinPath(key_info_dir_path, "done_" + std::to_string(i));
      FS_CHECK_OK(file_system_->FileExists(done_file_path));
    }
    int64_t undeletefiles, undeletedirs;
    FS_CHECK_OK(file_system_->DeleteRecursively(key_info_dir_path,
                                                &undeletefiles, &undeletedirs));
    // concat
    std::unique_ptr<fs::WritableFile> concat_file;
    FS_CHECK_OK(file_system_->NewWritableFile(concat_file_path, &concat_file));
    for (int32_t i = 0; i < part_num; ++i) {
      std::string file_path = JoinPath(sub_dir, std::to_string(i));
      FS_CHECK_OK(file_system_->FileExists(file_path));
      const uint64_t batch_size = 64 * 1024 * 1024;
      char* scratch = new char[batch_size];
      uint64_t offset = 0;
      std::unique_ptr<fs::RandomAccessFile> file;
      FS_CHECK_OK(file_system_->NewRandomAccessFile(file_path, &file));
      uint64_t file_size = 0;
      FS_CHECK_OK(file_system_->GetFileSize(file_path, &file_size));
      while (offset < file_size) {
        uint64_t n = std::min(batch_size, (file_size - offset));
        FS_CHECK_OK(file->Read(offset, n, scratch));
        FS_CHECK_OK(concat_file->Append(scratch, n));
        offset += n;
      }
      FS_CHECK_OK(file_system_->DeleteFile(file_path));
      free(scratch);
    }
    FS_CHECK_OK(concat_file->Close());
  }
}

std::unique_ptr<PersistentInStream> Snapshot::GetInStream(
    const std::string& key, size_t begin_pos) const {
  std::string file_path =
      JoinPath(root_path_, MakeValidFileName(key), concat_file_name_);
  PersistentInStream* ret =
      new PersistentInStream(file_system_, file_path, begin_pos);
  return std::unique_ptr<PersistentInStream>(ret);
}

std::unique_ptr<PersistentInStream> Snapshot::GetInStream(
    const std::string& key, int32_t part_id, int32_t part_num, int32_t dim_num,
    int64_t byte_size_of_each_dim) const {
  std::string file_path =
      JoinPath(root_path_, MakeValidFileName(key), concat_file_name_);
  uint64_t file_size = 0;
  FS_CHECK_OK(file_system_->GetFileSize(file_path, &file_size));
  CHECK_GT(file_size, 0);
  CHECK_EQ(file_size, dim_num * byte_size_of_each_dim);
  BalancedSplitter splitter = BalancedSplitter(dim_num, part_num);
  int64_t begin_pos = splitter.At(part_id).begin() * byte_size_of_each_dim;
  return GetInStream(key, begin_pos);
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(
    const std::string& key, int32_t part_id, int32_t part_num) {
  std::string dir_path = JoinPath(root_path_, MakeValidFileName(key));
  if (file_system_->IsDirectory(dir_path) == fs::Status::NOT_FOUND) {
    FS_CHECK_OK(file_system_->CreateDir(dir_path));
  }
  FS_CHECK_OK(file_system_->IsDirectory(dir_path));
  std::string key_info_dir_path = JoinPath(dir_path, key_info_dir_name_);
  if (file_system_->IsDirectory(key_info_dir_path) == fs::Status::NOT_FOUND) {
    FS_CHECK_OK(file_system_->CreateDir(key_info_dir_path));
  }
  FS_CHECK_OK(file_system_->IsDirectory(key_info_dir_path));
  if (part_id == 0) {
    std::unique_ptr<fs::WritableFile> part_num_file;
    std::string part_num_file_path =
        JoinPath(key_info_dir_path, "total_" + std::to_string(part_num));
    FS_CHECK_OK(
        file_system_->NewWritableFile(part_num_file_path, &part_num_file));
  }
  std::string file_path = JoinPath(dir_path, std::to_string(part_id));
  PersistentOutStream* ret = new PersistentOutStream(file_system_, file_path);
  return std::unique_ptr<PersistentOutStream>(ret);
}

void Snapshot::OnePartDone4Key(const std::string& key, const int32_t part_id) {
  std::string done_file_path =
      JoinPath(root_path_, MakeValidFileName(key), key_info_dir_name_,
               "done_" + std::to_string(part_id));
  CHECK(file_system_->FileExists(done_file_path) == fs::Status::NOT_FOUND);
  PersistentOutStream out_stream(file_system_, done_file_path);
}

}  // namespace oneflow
