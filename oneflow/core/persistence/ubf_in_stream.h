#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/persistence/ubf_item.h"
#include "oneflow/core/persistence/ubf_util.h"

namespace oneflow {

//  united binary formatted file input stream
class UbfInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfInStream);
  UbfInStream() = delete;
  UbfInStream(fs::FileSystem* fs, const std::string& file_path)
      : in_stream_(of_make_unique<NormalPersistentInStream>(fs, file_path)) {}
  virtual ~UbfInStream() = default;

  // 0: success
  // -1: eof
  int32_t ReadOneItem(UbfItem* item);

 protected:
  std::unique_ptr<NormalPersistentInStream>& mut_in_stream() {
    return in_stream_;
  }
  // 0: success
  // -1: eof
  virtual int32_t ReadDesc(char* s, size_t n) { return Read(s, n); }
  // 0: success
  // -1: eof
  virtual int32_t Read(char* s, size_t n) { return in_stream_->Read(s, n); }

 private:
  std::unique_ptr<NormalPersistentInStream> in_stream_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_
