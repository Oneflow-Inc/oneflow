#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_READER_H_

#include "oneflow/core/persistence/data_reader.h"

namespace oneflow {

class NormalDataReader final : public DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalDataReader);
  NormalDataReader() = delete;

  NormalDataReader(fs::FileSystem* fs, const std::string& file_path)
      : DataReader(fs, file_path) {
    LOG(INFO) << "New NormalDataReader " << file_path;
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_DATA_READER_H_
