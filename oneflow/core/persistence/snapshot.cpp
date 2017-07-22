#include "oneflow/core/persistence/snapshot.h"
#include "tensorflow/core/lib/io/path.h"

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
    std::string sub_dir = tensorflow::io::JoinPath(root_path_, sub_dir_name);
    TF_CHECK_OK(env_->IsDirectory(sub_dir));
    // for the children of the sub_dir
    std::vector<std::string> file_names;
    TF_CHECK_OK(env_->GetChildren(sub_dir, &file_names));
    CHECK_NE(file_names.size(), 0);
    std::string concat_file_path =
        tensorflow::io::JoinPath(sub_dir, concat_file_name_);
    // if the children number is 1 , the child must be a file
    // if the file name is not the concat_file_name_ , the file name must be
    // "0", and then replace the name to concat_file_name_
    if (file_names.size() == 1) {
      std::string file_path = tensorflow::io::JoinPath(sub_dir, file_names[0]);
      TF_CHECK_OK(env_->FileExists(file_path));
      if (file_names[0] != concat_file_name_) {
        CHECK_EQ(file_names[0], "0");
        TF_CHECK_OK(env_->RenameFile(file_path, concat_file_path));
      }
      continue;
    }
    // if the children number is more than 1, every child must be a file,
    // and the file name must be {0,1,2 ... n} , n is the part number
    // and then CONCAT the files to one file, delete origin files
    int32_t max_part_id = 0;
    for (std::string sub_file_name : file_names) {
      std::string file_path = tensorflow::io::JoinPath(sub_dir, sub_file_name);
      TF_CHECK_OK(env_->FileExists(file_path));
      int32_t part_id = oneflow_cast<int32_t>(sub_file_name);
      max_part_id = std::max(max_part_id, part_id);
    }
    CHECK_EQ(max_part_id, file_names.size() - 1);
    // concat
    std::unique_ptr<tensorflow::WritableFile> concat_file;
    TF_CHECK_OK(env_->NewWritableFile(concat_file_path, &concat_file));
    for (int32_t i = 0; i <= max_part_id; ++i) {
      std::string file_path =
          tensorflow::io::JoinPath(sub_dir, std::to_string(i));
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
    const std::string& key, size_t begin_pos) {
  std::string file_path = tensorflow::io::JoinPath(
      root_path_, MakeValidFileName(key), concat_file_name_);
  PersistentInStream* ret = new PersistentInStream(file_path, begin_pos);
  return std::unique_ptr<PersistentInStream>(ret);
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(
    const std::string& key, int32_t part_id) {
  std::string dir_path =
      tensorflow::io::JoinPath(root_path_, MakeValidFileName(key));
  if (env_->IsDirectory(dir_path).code() == tensorflow::error::NOT_FOUND) {
    TF_CHECK_OK(env_->CreateDir(dir_path));
  }
  TF_CHECK_OK(env_->IsDirectory(dir_path));
  std::string file_path =
      tensorflow::io::JoinPath(dir_path, std::to_string(part_id));
  PersistentOutStream* ret = new PersistentOutStream(file_path);
  return std::unique_ptr<PersistentOutStream>(ret);
}

}  // namespace oneflow
