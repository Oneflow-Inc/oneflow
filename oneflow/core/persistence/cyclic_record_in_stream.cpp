#include "oneflow/core/persistence/cyclic_record_in_stream.h"
namespace oneflow {

int32_t CyclicRecordInStream::ReadMeta(char* s, size_t n) {
  int ret = Read(s, n);
  if (!ret) { return ret; }

  //  this is eof, return to header by using new NormalPersistentInStream
  Init();
  return Read(s, n);
}

}  // namespace oneflow
