#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/data_set_format.h"
#include "oneflow/core/persistence/data_set_util.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

class DataSetInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSetInStream)
  DataSetInStream(fs::FileSystem* fs, const std::string& file_path)
      : fs_(fs),
        file_path_(file_path),
        header_(of_make_unique<DataSetHeader>()) {
    Init();
  }
  virtual ~DataSetInStream() = default;

  int32_t ReadRecord(std::unique_ptr<Record, decltype(&free)>* item);

  //	getter
  const DataSetHeader* header() const {
    CHECK(header_.get());
    return header_.get();
  }

 protected:
  virtual int32_t ReadMeta(char* s, size_t n) { return Read(s, n); }
  virtual int32_t Read(char* s, size_t n) { return in_stream_->Read(s, n); }
  void Init() {
    in_stream_ = of_make_unique<NormalPersistentInStream>(fs_, file_path_, 0);
    InitHeader();
  }

 private:
  void InitHeader();
  fs::FileSystem* fs_;
  std::string file_path_;
  std::unique_ptr<DataSetHeader> header_;
  std::unique_ptr<NormalPersistentInStream> in_stream_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
