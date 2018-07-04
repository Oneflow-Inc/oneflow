#ifndef ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class BinaryInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryInStream);
  virtual ~BinaryInStream() = default;

  // 0: success
  // -1: eof
  virtual int32_t Read(char* s, size_t n) = 0;

  virtual uint64_t file_size() const = 0;
  virtual uint64_t cur_file_pos() const = 0;
  virtual void set_cur_file_pos(uint64_t val) = 0;
  virtual bool IsEof() const = 0;

 protected:
  BinaryInStream() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_
