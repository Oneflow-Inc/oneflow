#include "thread/thread_manager.h"
#include <utility>

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  JoinAllThreads();
}

Thread* ThreadMgr::GetThrd(uint64_t thrd_loc_id) {
  return thrd_loc_id2thread_.at(thrd_loc_id).get();
}

void ThreadMgr::JoinAllThreads() {
  for (auto& pair : thrd_loc_id2thread_) {
    pair.second->Join();
  }
}

ThreadMgr::ThreadMgr() {
  TODO();
}

}  // namespace oneflow
