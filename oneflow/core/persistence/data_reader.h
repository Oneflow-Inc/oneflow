#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_

#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataReader);
  DataReader() = delete;
  virtual ~DataReader();

  // 0: success
  // -1: eof
  int32_t ReadLine(std::string* line);
  int32_t Read(char* s, size_t n);

 protected:
  DataReader(fs::FileSystem*, const std::string& file_path);
  virtual void AddNForCurFilePos(uint64_t n) { cur_file_pos_ += n; }
  uint64_t file_size() const { return file_size_; }
  uint64_t cur_file_pos() const { return cur_file_pos_; }
  void set_cur_file_pos(uint64_t val) { cur_file_pos_ = val; }

 private:
  bool IsEof() const;
  void UpdateBuffer();

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

#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_
