#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_

#include "oneflow/core/persistence/ubf_in_stream.h"

namespace oneflow {

class CyclicUbfInStream : public UbfInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicUbfInStream);
  CyclicUbfInStream() = delete;
  CyclicUbfInStream(fs::FileSystem* fs, const std::string& file_path)
      : UbfInStream(fs, file_path) {
    in_stream_resetter_ = [=]() {
      mut_in_stream().reset(new NormalPersistentInStream(fs, file_path, 0));
    };
  }

 protected:
  int32_t ReadDesc(char* s, size_t n) override;

 private:
  std::function<void()> in_stream_resetter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
