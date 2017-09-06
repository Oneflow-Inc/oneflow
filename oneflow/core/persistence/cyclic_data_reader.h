#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_

#include "oneflow/core/persistence/data_reader.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class CyclicDataReader final : public DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicDataReader);
  CyclicDataReader() = delete;
  ~CyclicDataReader();

  CyclicDataReader(const std::string& filepath);

  int32_t ReadLine(std::string* line) override;
  int32_t Read(char* s, size_t n) override { UNEXPECTED_RUN(); }

 private:
  void UpdateBuffer();

  // file
  std::unique_ptr<fs::RandomAccessFile> file_;
  uint64_t file_size_;
  uint64_t cur_file_pos_;
  // buffer
  char* buffer_;
  char* cur_buf_begin_;
  char* cur_buf_end_;
  static const size_t buffer_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
