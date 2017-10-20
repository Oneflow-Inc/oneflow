#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
#include "oneflow/core/persistence/record_in_stream.h"
namespace oneflow {

class NormalRecordInStream : public RecordInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalRecordInStream);
  NormalRecordInStream(fs::FileSystem* fs, const std::string& file_path)
      : RecordInStream(fs, file_path) {}
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
