#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_CIRCULAR_LINE_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_CIRCULAR_LINE_READER_H_

#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class PersistentCircularLineReader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentCircularLineReader);
  PersistentCircularLineReader() = delete;
  ~PersistentCircularLineReader() = default;

  PersistentCircularLineReader(const std::string& filepath);

  void ReadLine(std::string* line);

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_CIRCULAR_LINE_READER_H_
