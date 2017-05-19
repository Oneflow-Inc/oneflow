#ifndef ONEFLOW_ACTOR_THREAD_MANAGER_H_
#define ONEFLOW_ACTOR_THREAD_MANAGER_H_

#include <memory>
#include "actor/thread.h"
#include "actor/actor_msg_bus.h"
#include "common/channel.h"
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

  void JoinAllThreads();

  Channel<ActorMsg>* GetMsgChanFromThrdLocId(uint64_t thrd_loc_id);

 private:
  ThreadMgr() = default;

  HashMap<uint64_t, std::unique_ptr<Thread>> thrd_loc_id2thread_;
};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_THREAD_MANAGER_H_
