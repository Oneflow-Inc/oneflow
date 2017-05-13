#ifndef ONEFLOW_THREAD_COMM_BUS_H_
#define ONEFLOW_THREAD_COMM_BUS_H_

#include "common/blocking_queue.h"

namespace oneflow {

class CommBus final {
public:
  OF_DISALLOW_COPY_AND_MOVE(CommBus);
  ~CommBus() = default;

  static CommBus& Singleton() {
    static CommBus obj;
    return obj;
  }

  void InsertThrdLocIdMsgQPair(uint64_t thrd_loc_id, BlockingQueue* msg_queue);

  void SendMsg(const Message& msg);

private:
  void SendMsg(const Message& msg, uint64_t thrd_loc_id);

  CommBus() = default;
  HashMap<uint64_t, BlockingQueue*> thrd_loc_id2msg_queue_;

};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_COMM_BUS_H_
