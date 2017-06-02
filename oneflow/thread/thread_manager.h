#ifndef ONEFLOW_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_THREAD_THREAD_MANAGER_H_

#include <memory>
#include "oneflow/thread/thread.h"
#include "oneflow/actor/actor_msg_bus.h"
#include "oneflow/common/channel.h"
#include "oneflow/common/protobuf.h"

namespace oneflow {

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ~ThreadMgr();

  static ThreadMgr& Singleton() {
    static ThreadMgr obj;
    return obj;
  }
  
  Thread* GetThrd(uint64_t thrd_loc_id);

  void JoinAllThreads();

 private:
  ThreadMgr();

  std::vector<std::unique_ptr<Thread>> threads_;
};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_MANAGER_H_
