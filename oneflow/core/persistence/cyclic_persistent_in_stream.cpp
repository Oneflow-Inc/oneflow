#include "oneflow/core/persistence/cyclic_persistent_in_stream.h"

namespace oneflow {

CyclicPersistentInStream::CyclicPersistentInStream(fs::FileSystem* fs,
                                                   const std::string& file_path)
    : PersistentInStream(fs, file_path, 0) {
  LOG(INFO) << "New CyclicPersistentInStream " << file_path;
}

void CyclicPersistentInStream::AddNForCurFilePos(uint64_t n) {
  set_cur_file_pos((cur_file_pos() + n) % file_size());
}

}  // namespace oneflow
