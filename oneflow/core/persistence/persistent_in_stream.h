#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStream);
  virtual ~PersistentInStream() = default;

  // 0: success
  // -1: eof
  virtual int32_t ReadLine(std::string* l) = 0;
  virtual int32_t Read(char* s, size_t n) = 0;

 protected:
  PersistentInStream() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
