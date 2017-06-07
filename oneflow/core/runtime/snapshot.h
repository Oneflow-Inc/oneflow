#ifndef ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_
#define ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_

#include "tensorflow/core/platform/env.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class Snapshot final {
 public:
  class InStream;
  class OutStream;

  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;
  
  Snapshot(const std::string& snapshot_root_path);

  // Get Stream
  std::unique_ptr<InStream> GetInStream(const std::string& key,
                                        size_t begin_pos);
  std::unique_ptr<OutStream> GetOutStream(const std::string& key,
                                          int32_t part_id);

 private:
   // check the sub_dir of snapshot_root_path and files of sub_dir is legal
   // and concat the sub parallel file of the key
   void CheckAndConcat();

   // a uniform file name, this file is concated from 
   //   {part_0, part_1, ... part_n}
   static const char* concat_file_name_;
   std::string root_path_;
   tensorflow::Env* env_;
};

class Snapshot::InStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InStream);
  InStream() = delete;
  ~InStream() = default;

  template<typename T>
  InStream& operator >> (T& x) {
    if (!good()) {
      return *this;
    }
    CHECK(std::is_fundamental<T>::value);
    size_t n = sizeof(x);
    if (offset_ + n > file_size_) {
      is_eof_ = true;
      offset_ += n;
      return *this;
    }
    char* scratch = new char[n];
    tensorflow::StringPiece result;
    if (file_->Read(offset_, n, &result, scratch).code() != tensorflow::error::OK) {
      is_eof_ = true;
    }
    std::memcpy(&x, result.data(), result.size());
    offset_ += n;
    return *this;
  }

  InStream& Read(char* s, size_t n);

  bool good() const {
    return !is_eof_;
  }

  bool eof() const {
    return is_eof_;
  }

 private:
   friend class Snapshot;
   InStream(const std::string& file_path, uint64_t offset);
   std::unique_ptr<tensorflow::RandomAccessFile> file_;
   uint64_t file_size_;
   uint64_t offset_;
   bool is_eof_;
};

class Snapshot::OutStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStream);
  OutStream() = delete;
  ~OutStream() {
    file_->Close();
  }

  // the type T must be fundamental
  template<typename T>
  OutStream& operator << (const T& x) {
    const char* x_ptr = &x;
    CHECK(std::is_fundamental<T>::value);
    size_t n = sizeof(x);
    auto data = tensorflow::StringPiece(x_ptr, n);
    file_->Append(data);
    return *this;
  }

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  OutStream& Write(const char* s, size_t n);

 private:
   friend class Snapshot;
   OutStream(const std::string& file_path);
   std::unique_ptr<tensorflow::WritableFile> file_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_
