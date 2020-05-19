#ifndef ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_
#define ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_

namespace oneflow {

struct SizedBufferView {
  size_t capacity;  // allocated memory size for `data' field
  size_t size;      // valid data size
  char data[0];
};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_SIZED_BUFFER_VIEW_H_
