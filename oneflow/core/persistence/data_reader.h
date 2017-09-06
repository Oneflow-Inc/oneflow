#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataReader);
  virtual ~DataReader() = default;

  // 0: success
  // -1: eof
  virtual int32_t ReadLine(std::string* line) = 0;
  virtual int32_t Read(char* s, size_t n) = 0;

 protected:
  DataReader() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_READER_H_
