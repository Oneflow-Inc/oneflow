#include "oneflow/core/persistence/cyclic_data_reader.h"

namespace oneflow {

CyclicDataReader::CyclicDataReader(fs::FileSystem* fs,
                                   const std::string& file_path)
    : DataReader(fs, file_path) {
  LOG(INFO) << "New CyclicDataReader " << file_path;
}

void CyclicDataReader::AddNForCurFilePos(uint64_t n) {
  set_cur_file_pos((cur_file_pos() + n) % file_size());
}

}  // namespace oneflow
