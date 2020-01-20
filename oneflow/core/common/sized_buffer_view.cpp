#include "sized_buffer_view.h"

namespace oneflow {

SizedBufferView* SizedBufferView::PlacementNew(char* buffer, std::size_t buffer_size) {
  CHECK_GE(buffer_size, sizeof(SizedBufferView));
  auto* ret = new (buffer) SizedBufferView();
  ret->capacity = buffer_size - sizeof(SizedBufferView);
  return ret;
}

}  // namespace oneflow
