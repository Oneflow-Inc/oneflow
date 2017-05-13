#ifndef ONEFLOW_COMMON_BLOCKING_QUEUE_H_
#define ONEFLOW_COMMON_BLOCKING_QUEUE_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include "common/util.h"
//#include "actor/message.h"

namespace oneflow {

// TODO(liuguo): delete this line after Message is implement
class Message {
  public:
    uint64_t to_actor_id() const { return 0;}
};

class BlockingQueue final {
public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingQueue);
  BlockingQueue() = default;
  ~BlockingQueue() = default;

  int Write(const Message& msg);
  int Read(Message* msg);
  void Close();

private:
  std::queue<Message> msgs_;
  std::mutex mtx_;
  std::condition_variable write_cond_;
  std::condition_variable read_cond_;
  bool is_closed_;
};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_BLOCKING_QUEUE_H_
