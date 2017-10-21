#include "oneflow/core/persistence/cyclic_ofb_in_stream.h"
namespace oneflow {

int32_t CyclicOfbInStream::ReadMeta(char* s, size_t n) {
  int ret = Read(s, n);
  if (!ret) { return ret; }
  in_stream_resetter_();
  return Read(s, n);
}

}  // namespace oneflow
