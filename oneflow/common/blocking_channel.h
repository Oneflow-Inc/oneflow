#ifndef ONEFLOW_COMMON_BLOCKING_CHANNEL_H_
#define ONEFLOW_COMMON_BLOCKING_CHANNEL_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include "common/util.h"

namespace oneflow {

template<typename T>
class BlockingChannel final {
public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingChannel);
  BlockingChannel() = default;
  ~BlockingChannel() = default;

  int Write(const T& val);
  int Read(T* val);
  void Close();

private:
  std::queue<T> vals_;
  std::mutex mtx_;
  std::condition_variable write_cond_;
  std::condition_variable read_cond_;
  bool is_closed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_BLOCKING_CHANNEL_H_
