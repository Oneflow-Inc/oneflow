#ifndef ONEFLOW_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_THREAD_THREAD_MANAGER_H_

#include "thread/thread.h"
#include "thread/actor_msg_bus.h"
#include "common/blocking_channel.h"
#include "common/protobuf.h"

namespace oneflow {

class ThreadMgr final {
public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ~ThreadMgr();

  static ThreadMgr& Singleton() {
    static ThreadMgr obj;
    return obj;
  }

  void InitFromProto(const PbRpf<TaskProto>& tasks);

  void Join();

  BlockingChannel<ActorMsg>& GetMsgQ4ThrdWithThrdLocId(uint64_t thrd_loc_id);

private:
  ThreadMgr() = default;

  HashMap<uint64_t, std::unique_ptr<Thread>> thrd_loc_id2thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_MANAGER_H_
