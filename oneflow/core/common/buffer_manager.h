#ifndef ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_
#define ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

template<typename T>
class BufferMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BufferMgr);
  ~BufferMgr() = default;

  void NewChannel(const std::string& channel_name, size_t channel_size) {
    CHECK(name2channel_.emplace(channel_name, std::make_unique<Channel<T>>(channel_size)).second);
  }
  Channel<T>* Get(const std::string& channel_name) const {
    return name2channel_.at(channel_name).get();
  }

 private:
  friend class Global<BufferMgr>;
  BufferMgr() = default;

  HashMap<std::string, std::unique_ptr<Channel<T>>> name2channel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BUFFER_MANAGER_H_
