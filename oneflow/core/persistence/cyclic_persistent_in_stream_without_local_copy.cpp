#include "oneflow/core/persistence/cyclic_persistent_in_stream_without_local_copy.h"

namespace oneflow {

CyclicPersistentInStreamWithoutLocalCopy::
    CyclicPersistentInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                             const std::string& file_path)
    : PersistentInStreamWithoutLocalCopy(fs, file_path, 0) {
  LOG(INFO) << "New CyclicPersistentInStreamWithoutLocalCopy " << file_path;
  is_first_update_buffer_ = true;
}

void CyclicPersistentInStreamWithoutLocalCopy::UpdateBuffer() {
  if (is_first_update_buffer_ == false
      && file_size() <= mut_buffer()->size() - 1) {
    set_cur_buf_begin(mut_buffer()->data());
  } else {
    PersistentInStreamWithoutLocalCopy::UpdateBuffer();
  }
  is_first_update_buffer_ = false;
}

void CyclicPersistentInStreamWithoutLocalCopy::AddNForCurFilePos(uint64_t n) {
  set_cur_file_pos((cur_file_pos() + n) % file_size());
}

}  // namespace oneflow
