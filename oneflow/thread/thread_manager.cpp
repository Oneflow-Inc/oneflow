#include "thread/thread_manager.h"
#include <utility>
#include "thread/actor_msg_bus.h"
#include "runtime/runtime_info.h"

namespace oneflow {

Channel<ActorMsg>* ThreadMgr::GetMsgChanFromThrdLocId(uint64_t thrd_loc_id) {
  return thrd_loc_id2thread_.at(thrd_loc_id)->GetMsgChannelPtr();
}

void ThreadMgr::InitFromProto(const PbRpf<TaskProto>& tasks) {
  for (const TaskProto& task : tasks) {
    if (task.machine_id() != RuntimeInfo::Singleton().this_machine_id()) {
      continue;
    }
    if (thrd_loc_id2thread_.find(task.thrd_local_id()) ==
        thrd_loc_id2thread_.end()) {
      auto thrd = of_make_unique<Thread>();
      thrd->set_thrd_loc_id(task.thrd_local_id());
      CHECK(thrd_loc_id2thread_.emplace(
            task.thrd_local_id(), std::move(thrd)).second);
    }
    std::unique_ptr<Thread>& thrd =
      thrd_loc_id2thread_.at(task.thrd_local_id());
    thrd->AddActor(task);
  }
}

void ThreadMgr::JoinAllThreads() {
  for (auto& pair : thrd_loc_id2thread_) {
    pair.second->Join();
  }
}

ThreadMgr::~ThreadMgr() {
  JoinAllThreads();
}

}  // namespace oneflow
