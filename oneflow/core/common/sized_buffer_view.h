#ifndef ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_
#define ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_

#include <glog/logging.h>

namespace oneflow {

struct SizedBufferView {
  size_t capacity = 4096; // allocated memory size for `data' field
  size_t size; // valid data size
  char data[0];

  static SizedBufferView* PlacementNew(char* buffer, std::size_t buffer_size) {
    CHECK_GE(buffer_size, sizeof(SizedBufferView));
    auto* ret = new (buffer) SizedBufferView;
    ret -> capacity = buffer_size - sizeof(SizedBufferView);
    return ret;
  }

  private:
    SizedBufferView() {};
};

}

#endif  // ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_
