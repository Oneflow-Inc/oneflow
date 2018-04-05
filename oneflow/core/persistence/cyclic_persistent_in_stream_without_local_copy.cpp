#include "oneflow/core/persistence/cyclic_persistent_in_stream_without_local_copy.h"

namespace oneflow {

CyclicPersistentInStreamWithoutLocalCopy::
    CyclicPersistentInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                             const std::string& file_path)
    : PersistentInStreamWithoutLocalCopy(fs, file_path, 0) {
  LOG(INFO) << "New CyclicPersistentInStreamWithoutLocalCopy " << file_path;
}

void CyclicPersistentInStreamWithoutLocalCopy::AddNForCurFilePos(uint64_t n) {
  set_cur_file_pos((cur_file_pos() + n) % file_size());
}

}  // namespace oneflow
