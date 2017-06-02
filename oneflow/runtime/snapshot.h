#ifndef ONEFLOW_RUNTIME_SNAPSHOT_H_
#define ONEFLOW_RUNTIME_SNAPSHOT_H_

#include "common/util.h"
#include "tensorflow/core/platform/env.h"

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
   // check the sub_dir of the snapshot_root_path named key is exist
   // and concat the sub parallel file of the key
   void CheckAndConcat(const std::string& key);

   // return a uniform file name, this file is concated from 
   //   {part_0, part_1, ... part_n}
   static const std::string& ConcatFileName() {
     static std::string obj = "all";
     return obj;
   }

   std::string root_path_;
   tensorflow::Env* env_;
};

class Snapshot::InStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InStream);
  InStream() = delete;

  ~InStream() = default;

  template<typename T>
  InStream& operator >> (T& x);

  InStream& Read(char* s, size_t n);

  bool good() const {
    return !eofbit;
  }

  bool eof() const {
    return eofbit;
  }

 private:
   friend Snapshot;
   InStream(const std::string& file_path, uint64_t offset);
   std::unique_ptr<tensorflow::RandomAccessFile> file_;
   uint64_t file_size_;
   uint64_t offset_;
   bool eofbit;
};

class Snapshot::OutStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStream);
  OutStream() = delete;
  
  ~OutStream() {
    Close();
  }

  // the type T must be fundamental
  template<typename T>
  OutStream& operator << (const T& x);

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  OutStream& Write(const char* s, size_t n);

  tensorflow::Status Close() {
    return file_->Close();
  }

 private:
   friend Snapshot;
   OutStream(const std::string& file_path);
   std::unique_ptr<tensorflow::WritableFile> file_;
};

} // namespace oneflow

#endif // ONEFLOW_RUNTIME_SNAPSHOT_H_
