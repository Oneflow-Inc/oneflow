#include "oneflow/core/persistence/cyclic_data_set_in_stream.h"
namespace oneflow {

int32_t CyclicDataSetInStream::ReadMeta(char* s, size_t n) {
  int ret = Read(s, n);
  if (!ret) { return ret; }

  //  this is eof, return to header by using new NormalPersistentInStream
  Init();
  return Read(s, n);
}

}  // namespace oneflow
