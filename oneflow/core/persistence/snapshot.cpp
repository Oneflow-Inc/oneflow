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
  env_ = tensorflow::Env::Default();
  TF_CHECK_OK(env_->IsDirectory(snapshot_root_path));
  root_path_ = snapshot_root_path;
  CheckAndConcat();
}

void Snapshot::CheckAndConcat() {
  // the children of the root path must be dir, not file
  std::vector<std::string> sub_dir_names;
  TF_CHECK_OK(env_->GetChildren(root_path_, &sub_dir_names));
  for (std::string sub_dir_name : sub_dir_names) {
    std::string sub_dir = JoinPath(root_path_, sub_dir_name);
    TF_CHECK_OK(env_->IsDirectory(sub_dir));
    // for the children of the sub_dir
    std::vector<std::string> file_names;
    TF_CHECK_OK(env_->GetChildren(sub_dir, &file_names));
    CHECK_NE(file_names.size(), 0);
    std::string concat_file_path = JoinPath(sub_dir, concat_file_name_);
    // for condition after concat
    // if the children number is 1 , the child must be the concated file named
    // "all"
    if (file_names.size() == 1) {
      std::string file_path = JoinPath(sub_dir, file_names[0]);
      TF_CHECK_OK(env_->FileExists(file_path));
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
    TF_CHECK_OK(env_->IsDirectory(key_info_dir_path));
    std::vector<std::string> key_info_subs;
    TF_CHECK_OK(env_->GetChildren(key_info_dir_path, &key_info_subs));
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
      TF_CHECK_OK(env_->FileExists(done_file_path));
    }
    tensorflow::int64 undeletefiles, undeletedirs;
    TF_CHECK_OK(env_->DeleteRecursively(key_info_dir_path, &undeletefiles,
                                        &undeletedirs));
    // concat
    std::unique_ptr<tensorflow::WritableFile> concat_file;
    TF_CHECK_OK(env_->NewWritableFile(concat_file_path, &concat_file));
    for (int32_t i = 0; i < part_num; ++i) {
      std::string file_path = JoinPath(sub_dir, std::to_string(i));
      TF_CHECK_OK(env_->FileExists(file_path));
      const tensorflow::uint64 batch_size = 64 * 1024 * 1024;
      char* scratch = new char[batch_size];
      tensorflow::uint64 offset = 0;
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      TF_CHECK_OK(env_->NewRandomAccessFile(file_path, &file));
      tensorflow::uint64 file_size = 0;
      TF_CHECK_OK(env_->GetFileSize(file_path, &file_size));
      while (offset < file_size) {
        tensorflow::StringPiece data;
        tensorflow::uint64 n = std::min(batch_size, (file_size - offset));
        TF_CHECK_OK(file->Read(offset, n, &data, scratch));
        TF_CHECK_OK(concat_file->Append(data));
        offset += n;
      }
      TF_CHECK_OK(env_->DeleteFile(file_path));
      free(scratch);
    }
    TF_CHECK_OK(concat_file->Close());
  }
}

std::unique_ptr<PersistentInStream> Snapshot::GetInStream(
    const std::string& key, size_t begin_pos) const {
  std::string file_path =
      JoinPath(root_path_, MakeValidFileName(key), concat_file_name_);
  PersistentInStream* ret = new PersistentInStream(file_path, begin_pos);
  return std::unique_ptr<PersistentInStream>(ret);
}

std::unique_ptr<PersistentInStream> Snapshot::GetInStream(
    const std::string& key, int32_t part_id, int32_t part_num, int32_t dim_num,
    int64_t byte_size_of_each_dim) const {
  std::string file_path =
      JoinPath(root_path_, MakeValidFileName(key), concat_file_name_);
  tensorflow::uint64 file_size = 0;
  TF_CHECK_OK(env_->GetFileSize(file_path, &file_size));
  CHECK_GT(file_size, 0);
  CHECK_EQ(file_size, dim_num * byte_size_of_each_dim);
  BalancedSplitter splitter = BalancedSplitter(dim_num, part_num);
  int64_t begin_pos = splitter.At(part_id).begin() * byte_size_of_each_dim;
  return GetInStream(key, begin_pos);
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(
    const std::string& key, int32_t part_id, int32_t part_num) {
  std::string dir_path = JoinPath(root_path_, MakeValidFileName(key));
  if (env_->IsDirectory(dir_path).code() == tensorflow::error::NOT_FOUND) {
    TF_CHECK_OK(env_->CreateDir(dir_path));
  }
  TF_CHECK_OK(env_->IsDirectory(dir_path));
  std::string key_info_dir_path = JoinPath(dir_path, key_info_dir_name_);
  if (env_->IsDirectory(key_info_dir_path).code()
      == tensorflow::error::NOT_FOUND) {
    TF_CHECK_OK(env_->CreateDir(key_info_dir_path));
  }
  TF_CHECK_OK(env_->IsDirectory(key_info_dir_path));
  if (part_id == 0) {
    std::unique_ptr<tensorflow::WritableFile> part_num_file;
    std::string part_num_file_path =
        JoinPath(key_info_dir_path, "total_" + std::to_string(part_num));
    TF_CHECK_OK(env_->NewWritableFile(part_num_file_path, &part_num_file));
  }
  std::string file_path = JoinPath(dir_path, std::to_string(part_id));
  PersistentOutStream* ret = new PersistentOutStream(file_path);
  return std::unique_ptr<PersistentOutStream>(ret);
}

void Snapshot::OnePartDone4Key(const std::string& key, const int32_t part_id) {
  std::string done_file_path =
      JoinPath(root_path_, MakeValidFileName(key), key_info_dir_name_,
               "done_" + std::to_string(part_id));
  CHECK(env_->FileExists(done_file_path).code()
        == tensorflow::error::NOT_FOUND);
  PersistentOutStream out_stream(done_file_path);
}

}  // namespace oneflow
