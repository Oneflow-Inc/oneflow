#ifndef ONEFLOW_CORE_KERNEL_NORMAL_DATA_SET_IN_STREAM_H_
#define ONEFLOW_CORE_KERNEL_NORMAL_DATA_SET_IN_STREAM_H_
#include "oneflow/core/kernel/data_set_in_stream.h"
namespace oneflow {

class NormalDataSetInStream : public DataSetInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalDataSetInStream);
  NormalDataSetInStream(fs::FileSystem* fs, const std::string& file_path)
      : DataSetInStream(fs, file_path) {}
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_NORMAL_DATA_SET_IN_STREAM_H_
