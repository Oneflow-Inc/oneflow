#ifndef ONEFLOW_COMMON_BLOCKING_QUEUE_H_
#define ONEFLOW_COMMON_BLOCKING_QUEUE_H_

#include <queue>
#include "common/util.h"
#include "actor/message.h"

namespace enn {

// TODO(liuguo): delete this line after Message is implement
class Message;

class BlockingQueue {
public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingQueue);
  BlockingQueue() = default;
  ~BlockingQueue() = default;

  bool Write(const Message& msg);
  bool Read(Message* msg);
  void Close();

private:
  std::queue<Message> msgs_;
  std::mutex mtx_;
  std::condition_variable write_cond_;
  std::condition_variable read_cond_;
  bool is_closed_;
};

}  // namespace enn
