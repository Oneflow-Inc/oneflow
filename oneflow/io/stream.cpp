#include "io/stream.h"

#include <mutex>
#include <string>

namespace caffe {
namespace io {
Stream *Stream::CreateForRead(const std::string &path, bool try_create) {
  return io::FileSystem::
    GetInstance(path)->OpenForRead(path, try_create);
}

class FileStream : public Stream {
public:
  explicit FileStream(FILE *fp, bool use_stdio)
    : file_(fp), use_stdio_(use_stdio), valid_(true) {}

  virtual ~FileStream() {
    this->Done();
  }

  void ReadOneStep() override {
    if (std::fread(&key_len_, sizeof(int), 1, file_) == 0) {
      LOG(INFO) << "EOF reached, setting valid to false";
      valid_ = false;
      return;
    }
    CHECK_EQ(std::fread(&value_len_, sizeof(int), 1, file_), 1);
    CHECK_GT(key_len_, 0);
    CHECK_GT(value_len_, 0);
    if (key_len_ > key_.size()) {
      key_.resize(key_len_);
    }
    if (value_len_ > value_.size()) {
      value_.resize(value_len_);
    }
    CHECK_EQ(
      std::fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
    CHECK_EQ(
      std::fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
  }

  std::string key() override {
    CHECK(valid_) << "Reader is at invalid location!";
    return std::string(key_.data(), key_len_);
  }

  std::string value() override {
    CHECK(valid_) << "Reader is at invalid location!";
    return std::string(value_.data(), value_len_);
  }

  bool Valid() override { return valid_; }

  void Put(const std::string& key, const std::string& value) override {
    int key_len = key.size();
    int value_len = value.size();
    CHECK_EQ(std::fwrite(&key_len, sizeof(int), 1, file_), 1);
    CHECK_EQ(std::fwrite(&value_len, sizeof(int), 1, file_), 1);
    CHECK_EQ(
      std::fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
    CHECK_EQ(
      std::fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
  }

  void Seek(size_t pos) {
    // fseek don't test for eof, and clears the eof flag.
    std::fseek(file_, static_cast<long>(pos), SEEK_SET);
  }

  size_t Tell(void) {
    return std::ftell(file_);
  }

  bool AtEnd(void) const {
    return std::feof(file_) != 0;
  }

  void Done() override {
    if (file_ != nullptr && !use_stdio_) {
      CHECK_EQ(std::fclose(file_), 0);
      file_ = nullptr;
    }
  }

private:
  std::FILE *file_;
  bool use_stdio_;
  // std::lock_guard<std::mutex> lock_;
  bool valid_;
  int key_len_;
  std::vector<char> key_;
  int value_len_;
  std::vector<char> value_;
};

FileSystem *FileSystem::GetInstance(const std::string &path) {
  return LocalFileSystem::GetInstance();
}

Stream *LocalFileSystem::Open(const std::string &path,
  const char* const mode,
  bool allow_null) {
  bool use_stdio = false;
  FILE *fp = NULL;
  const char *fname = path.c_str();

  if (!use_stdio) {
    std::string flag = mode;
    if (flag == "w") flag = "ab+";
    if (flag == "r") flag = "rb";
    fp = std::fopen(fname, flag.c_str());
  }
  if (fp != NULL) {
    return new FileStream(fp, use_stdio);
  }
  else {
    CHECK(allow_null) << " LocalFileSystem: fail to open \"" << path << '\"';
    return NULL;
  }
}

Stream *LocalFileSystem::OpenForRead(const std::string &path, bool allow_null) {
  return Open(path, "r", allow_null);
}
Stream *LocalFileSystem::OpenForWrite(const std::string &path, bool allow_null) {
  return Open(path, "w", allow_null);
}
}
}