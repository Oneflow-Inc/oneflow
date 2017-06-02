#include "runtime/snapshot.h"
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "common/util.h"

namespace oneflow {

Snapshot::Snapshot(const std::string& snapshot_root_path) {
  env_ = tensorflow::Env::Default();
  TF_CHECK_OK(env_->IsDirectory(snapshot_root_path));
  root_path_ = snapshot_root_path;
}

void Snapshot::CheckAndConcat(const std::string& key) {
  // the key of the root path must be dir, not file
  std::string sub_dir = tensorflow::io::JoinPath(root_path_, key);
  TF_CHECK_OK(env_->IsDirectory(sub_dir));
  // for the children of the sub_dir named key
  std::vector<std::string> file_names;
  env_->GetChildren(sub_dir, &file_names);
  CHECK_NE(file_names.size(), 0);
  std::string concat_file_path = tensorflow::io::JoinPath(sub_dir, ConcatFileName());
  // if the children number is 1 , the child must be a file
  // if the file name is not the ConcatFileName() , the file name must be "0",
  // and then replace the name to ConcatFileName()
  if (file_names.size() == 1) {
    std::string file_path = tensorflow::io::JoinPath(sub_dir, file_names[0]);
    TF_CHECK_OK(env_->FileExists(file_path));
    if (file_names[0] != ConcatFileName()) {
      CHECK_EQ(file_names[0], "0");
      env_->RenameFile(file_path, concat_file_path);
    }
    return;
  }
  // if the children number is more than 1, every child must be a file,
  // and the file name must be {0,1,2 ... n} , n is the part number
  // and then CONCAT the files to one file, delete origin files
  uint64_t max_part_id = 0;
  for (std::string sub_file_name : file_names) {
    std::string file_path = tensorflow::io::JoinPath(sub_dir, sub_file_name);
    TF_CHECK_OK(env_->FileExists(file_path));
    uint64_t part_id = StoullOrDie(sub_file_name);
    max_part_id = std::max(max_part_id, part_id);
  }
  CHECK_EQ(max_part_id, file_names.size() - 1);
  // concat
  std::unique_ptr<tensorflow::WritableFile> concat_file;
  TF_CHECK_OK(env_->NewWritableFile(concat_file_path, &concat_file));
  for (uint64_t i = 0; i <= max_part_id; i++) {
    std::string file_path = tensorflow::io::JoinPath(sub_dir,
      std::to_string(i));
    std::string* data_ptr;
    TF_CHECK_OK(tensorflow::ReadFileToString(env_, file_path, data_ptr));
    TF_CHECK_OK(concat_file->Append(tensorflow::StringPiece(*data_ptr)));
    TF_CHECK_OK(env_->DeleteFile(file_path));
  }
  concat_file->Close();
}

std::unique_ptr<Snapshot::InStream> Snapshot::GetInStream(
    const std::string& key,
    size_t begin_pos) {
  CheckAndConcat(key);
  std::string file_path = tensorflow::io::JoinPath(root_path_, key,
                                                   ConcatFileName());
  InStream* ret = new InStream(file_path, begin_pos);
  return std::unique_ptr<InStream>(ret);
}

std::unique_ptr<Snapshot::OutStream> Snapshot::GetOutStream(
    const std::string& key,
    int32_t part_id) {
  std::string dir_path = tensorflow::io::JoinPath(root_path_, key);
  if (env_->IsDirectory(dir_path).code() == tensorflow::error::NOT_FOUND) {
    env_->CreateDir(dir_path);
  }
  TF_CHECK_OK(env_->IsDirectory(dir_path));
  std::string file_path = tensorflow::io::JoinPath(dir_path,
                                                   std::to_string(part_id));
  OutStream* ret = new OutStream(file_path);
  return std::unique_ptr<OutStream>(ret);
}

Snapshot::InStream::InStream(const std::string& file_path, uint64_t offset) {
  tensorflow::Env* env_ = tensorflow::Env::Default();
  TF_CHECK_OK(env_->FileExists(file_path));
  TF_CHECK_OK(env_->NewRandomAccessFile(file_path, &file_));
  offset_ = offset;
  env_->GetFileSize(file_path, &file_size_);
  if (offset < file_size_) {
    eofbit = false;
  } else {
    eofbit = true;
  }
}

template<typename T>
Snapshot::InStream& Snapshot::InStream::operator >> (T& x) {
  if (eof()) {
    return *this;
  }
  CHECK(std::is_fundamental<T>);
  size_t n = sizeof(x);
  if (offset_ + n > file_size_) {
    eofbit = true;
    offset_ += n;
    return *this;
  }
  char* scratch = new char[size_t];
  tensorflow::StringPiece result;
  if (file_->Read(offset_, n, &result, scratch).code() != tensorflow::error::OK) {
    eofbit = true;
  }
  std::memcpy(&x, result.data(), result.size());
  offset_ += n;
  return *this;
}

Snapshot::InStream& Snapshot::InStream::Read(char* s, size_t n) {
  if (eof()) {
    return *this;
  }
  if (offset_ + n > file_size_) {
    eofbit = true;
    offset_ += n;
    return *this;
  };
  tensorflow::StringPiece result;
  char* scratch = new char[n];
  if (file_->Read(offset_, n, &result, scratch).code() != tensorflow::error::OK) {
    eofbit = true;
  }
  std::memcpy(s, result.data(), result.size());
  offset_ += n;
  return *this;
}

Snapshot::OutStream::OutStream(const std::string& file_path) {
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(file_path, &file_));
}

template<typename T>
Snapshot::OutStream& Snapshot::OutStream::operator << (const T& x) {
  const char* x_ptr = &x;
  CHECK(std::is_fundamental<T>);
  size_t n = sizeof(x);
  auto data = tensorflow::StringPiece(x_ptr, n);
  file_->Append(data);
  return *this;
}

Snapshot::OutStream& Snapshot::OutStream::Write(const char* s, size_t n) {
  auto data = tensorflow::StringPiece(s, n);
  file_->Append(data);
  return *this;
}

}  // namespace oneflow
