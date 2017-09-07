#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_

#include "oneflow/core/persistence/data_reader.h"

namespace oneflow {

class CyclicDataReader final : public DataReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicDataReader);
  CyclicDataReader() = delete;
  ~CyclicDataReader() = default;

  CyclicDataReader(fs::FileSystem* fs, const std::string& file_path);

  void AddNForCurFilePos(uint64_t n) override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_DATA_READER_H_
