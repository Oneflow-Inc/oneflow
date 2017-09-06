#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_

#include "oneflow/core/persistence/data_reader.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/env.h"

namespace oneflow {

class CyclicDataReader final : public DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicDataReader);
  CyclicDataReader() = delete;
  ~CyclicDataReader() = default;

  CyclicDataReader(const std::string& filepath);

  int32_t ReadLine(std::string* line) override;
  int32_t Read(char* s, size_t n) override { UNEXPECTED_RUN(); }

 private:
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::InputBuffer> in_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
