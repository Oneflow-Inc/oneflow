#ifndef ONEFLOW_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_THREAD_THREAD_MANAGER_H_

#include "thread/thread.h"
#include "common/proto_io.h"

namespace oneflow {

class ThreadMgr {
public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ~ThreadMgr() = default;

  static ThreadMgr& Singleton() {
    static ThreadMgr obj;
    return obj;
  }

  void InitFromProto(const PbRpf<TaskProto>& tasks);

private:
  ThreadMgr() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_THREAD_THREAD_MANAGER_H_
