#ifndef ONEFLOW_CORE_COMMON_BUFFER_H_
#define ONEFLOW_CORE_COMMON_BUFFER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
class Buffer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Buffer);
  explicit Buffer(size_t len) : len_(len) {}
  ~Buffer() = default;

  void Send(const T& val) { TODO(); }
  T Receive() { TODO(); }

 private:
  size_t len_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BUFFER_H_
