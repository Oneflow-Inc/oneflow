#ifndef ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_
#define ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/buffer.h"

namespace oneflow {

template<typename T>
class BufferMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BufferMgr);
  ~BufferMgr() = default;

  void NewBuffer(const std::string& buffer_name, size_t buffer_size) {
    CHECK(name2buffer_.emplace(buffer_name, std::make_unique<Buffer<T>>(buffer_size)).second);
  }
  Buffer<T>* Get(const std::string& buffer_name) const {
    return name2buffer_.at(buffer_name).get();
  }

 private:
  friend class Global<BufferMgr>;
  BufferMgr() = default;

  HashMap<std::string, std::unique_ptr<Buffer<T>>> name2buffer_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_
