#ifndef ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/persistence/ubf_header.h"
#include "oneflow/core/persistence/ubf_item.h"
#include "oneflow/core/persistence/ubf_util.h"

namespace oneflow {

//  oneflow binary file input stream
class UbfInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfInStream);
  UbfInStream() = delete;
  UbfInStream(fs::FileSystem* fs, const std::string& file_path)
      : header_(of_make_unique<UbfHeader>()),
        in_stream_(of_make_unique<NormalPersistentInStream>(fs, file_path, 0)) {
    ResetHeader();
  }
  virtual ~UbfInStream() = default;

  int32_t ReadOneItem(std::unique_ptr<UbfItem, decltype(&free)>* item);

  //	getter
  const UbfHeader* header() const {
    CHECK(header_);
    return header_.get();
  }

 protected:
  std::unique_ptr<NormalPersistentInStream>& mut_in_stream() {
    return in_stream_;
  }
  void ResetHeader();
  virtual int32_t ReadMeta(char* s, size_t n) { return Read(s, n); }
  virtual int32_t Read(char* s, size_t n) { return in_stream_->Read(s, n); }

 private:
  std::unique_ptr<UbfHeader> header_;
  std::unique_ptr<NormalPersistentInStream> in_stream_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_DATA_SET_IN_STREAM_H_
