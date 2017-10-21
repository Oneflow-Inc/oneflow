#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_

#include "oneflow/core/persistence/ofb_in_stream.h"
namespace oneflow {

class CyclicOfbInStream : public OfbInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicOfbInStream);
  CyclicOfbInStream() = delete;
  CyclicOfbInStream(fs::FileSystem* fs, const std::string& file_path)
      : OfbInStream(fs, file_path) {
    in_stream_resetter_ = [=]() {
      mut_in_stream().reset(new NormalPersistentInStream(fs, file_path, 0));
      ResetHeader();
    };
  }

 protected:
  int32_t ReadMeta(char* s, size_t n) override;

 private:
  std::function<void()> in_stream_resetter_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_SET_IN_STREAM_H_
