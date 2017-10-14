#ifndef ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/data_set_format.h"
#include "oneflow/core/kernel/data_set_util.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class DataSetInStream : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataSetInStream)
  DataSetInStream(fs::FileSystem* fs, const std::string& file_path)
      : PersistentInStream(fs, file_path, 0),
        header_(of_make_unique<DataSetHeader>()),
        label_desc_(DataSetUtil::Malloc<DataSetLabelDesc>(1)) {
    Init();
  }
  virtual ~DataSetInStream() = default;
  virtual void AddNForCurFilePos(uint64_t n) override {
    set_cur_file_pos(cur_file_pos() + n);
  }

  int32_t ReadDataItem(std::unique_ptr<DataItem, decltype(&free)>* item);

  //	getter
  const DataSetHeader* header() const {
    CHECK(header_.get());
    return header_.get();
  }
  const DataSetLabelDesc* label_desc() const {
    CHECK(label_desc_.get());
    return label_desc_.get();
  }

 protected:
 private:
  void Init() {
    InitHeader();
    ReadLabelDesc();
  }
  void InitHeader();
  void ReadLabelDesc();
  std::unique_ptr<DataSetHeader> header_;
  std::unique_ptr<DataSetLabelDesc, decltype(&free)> label_desc_;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_DATA_SET_IN_STREAM_H_
