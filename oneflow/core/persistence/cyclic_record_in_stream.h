#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_

#include "oneflow/core/persistence/record_in_stream.h"
namespace oneflow {

class CyclicRecordInStream : public RecordInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicRecordInStream);
  CyclicRecordInStream(fs::FileSystem* fs, const std::string& file_path)
      : RecordInStream(fs, file_path) {}

 protected:
  int32_t ReadMeta(char* s, size_t n) override;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
