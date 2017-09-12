#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStream);
  PersistentInStream() = delete;
  virtual ~PersistentInStream();

  // 0: success
  // -1: eof
  int32_t ReadLine(std::string* line);
  int32_t Read(char* s, size_t n);

 protected:
  PersistentInStream(fs::FileSystem*, const std::string& file_path,
                     uint64_t offset);
  virtual void AddNForCurFilePos(uint64_t n) { cur_file_pos_ += n; }
  uint64_t file_size() const { return file_size_; }
  uint64_t cur_file_pos() const { return cur_file_pos_; }
  void set_cur_file_pos(uint64_t val) { cur_file_pos_ = val; }

 private:
  void UpdateBuffer();
  bool IsEof() const;

  // file
  std::unique_ptr<fs::RandomAccessFile> file_;
  uint64_t file_size_;
  uint64_t cur_file_pos_;
  // buffer
  char* buffer_;
  char* cur_buf_begin_;
  char* cur_buf_end_;
  static const uint64_t buffer_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
