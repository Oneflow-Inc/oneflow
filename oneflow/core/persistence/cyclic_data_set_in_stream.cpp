#include "oneflow/core/persistence/cyclic_data_set_in_stream.h"
namespace oneflow {

void CyclicDataSetInStream::AddNForCurFilePos(uint64_t n) {
  uint64_t pos = cur_file_pos() + n;
  if (pos >= file_size()) { pos = (pos + sizeof(*header())) % file_size(); }
  set_cur_file_pos(pos);
}

}  // namespace oneflow
