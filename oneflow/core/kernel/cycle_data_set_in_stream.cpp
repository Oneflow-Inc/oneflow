#include "oneflow/core/kernel/cycle_data_set_in_stream.h"
namespace oneflow {

void CycleDataSetInStream::AddNForCurFilePos(uint64_t n) {
  CHECK(!(n % FlexibleSizeOf<DataItem>(header()->TensorElemCount())));
  uint64_t pos = cur_file_pos() + n;
  if (pos > file_size()) {
    set_cur_file_pos((pos + header()->DataBodyOffset()) % file_size());
  } else {
    set_cur_file_pos(pos);
  }
}

}  // namespace oneflow
