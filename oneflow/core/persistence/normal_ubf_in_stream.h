#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
#include "oneflow/core/persistence/ubf_in_stream.h"
namespace oneflow {

class NormalUbfInStream : public UbfInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalUbfInStream);
  NormalUbfInStream() = delete;
  NormalUbfInStream(fs::FileSystem* fs, const std::string& file_path)
      : UbfInStream(fs, file_path) {}
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_SET_IN_STREAM_H_
