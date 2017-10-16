#ifndef ONEFLOW_CORE_KERNEL_CYCLIC_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_KERNEL_CYCLIC_DATA_SET_IN_STREAM_H_

#include "oneflow/core/persistence/data_set_in_stream.h"
namespace oneflow {

class CyclicDataSetInStream : public DataSetInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicDataSetInStream);
  CyclicDataSetInStream(fs::FileSystem* fs, const std::string& file_path)
      : DataSetInStream(fs, file_path) {}

  virtual void AddNForCurFilePos(uint64_t n) override;
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_CYCLIC_DATA_SET_IN_STREAM_H_
