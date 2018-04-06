#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITH_LOCAL_COPY_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITH_LOCAL_COPY_H_

#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class CyclicPersistentInStreamWithLocalCopy final : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicPersistentInStreamWithLocalCopy);
  CyclicPersistentInStreamWithLocalCopy() = delete;
  ~CyclicPersistentInStreamWithLocalCopy() = default;

  CyclicPersistentInStreamWithLocalCopy(fs::FileSystem* fs,
                                        const std::string& file_path);

  int32_t ReadLine(std::string* l) override {
    return (this->*read_line_mthd_)(l);
  }
  int32_t Read(char* s, size_t n) override { return (this->*read_mthd_)(s, n); }

 private:
  int32_t ReadLineAndWriteToLocal(std::string* l);
  int32_t ReadAndWriteToLocal(char* s, size_t n);
  int32_t ReadLineFromLocal(std::string* l) { return in_stream_->ReadLine(l); }
  int32_t ReadFromLocal(char* s, size_t n) { return in_stream_->Read(s, n); }

  void CopyToLocalFinish();

  std::unique_ptr<PersistentInStream> in_stream_;
  std::string local_copy_path_;
  std::unique_ptr<PersistentOutStream> out_stream_;
  int32_t (CyclicPersistentInStreamWithLocalCopy::*read_line_mthd_)(
      std::string*);
  int32_t (CyclicPersistentInStreamWithLocalCopy::*read_mthd_)(char*, size_t);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITH_LOCAL_COPY_H_
