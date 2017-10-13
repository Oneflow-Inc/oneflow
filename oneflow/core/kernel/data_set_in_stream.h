#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/data_set_format.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class DataSetInStream : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSetInStream)
  DataSetInStream(fs::FileSystem* fs, const std::string& file_path)
      : PersistentInStream(fs, file_path, 0),
        header_(of_make_unique<DataSetHeader>()) {
    Init();
  }
  virtual ~DataSetInStream() = default;
  virtual void AddNForCurFilePos(uint64_t n) override {
    CHECK(!(n % FlexibleSizeOf<DataItem>(header()->TensorElemCount())));
    set_cur_file_pos(cur_file_pos() + n);
  }

  int32_t ReadDataItem(std::unique_ptr<DataItem>* item);

 protected:
  //	getter
  const DataSetHeader* header() const { return header_.get(); }

 private:
  void Init() {
    InitHeader();
    SkipLabelDesc();
  }
  void InitHeader();
  void SkipLabelDesc();
  std::unique_ptr<DataSetHeader> header_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
