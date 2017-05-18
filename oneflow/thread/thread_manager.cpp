#include "thread/thread_manager.h"
#include "thread/actor_msg_bus.h"
#include "runtime/runtime_info.h"

namespace oneflow{

BlockingChannel<ActorMsg>* ThreadMgr::GetMsgChannelPtr4ThrdWithThrdLocId(
    uint64_t thrd_loc_id) {
  return thrd_loc_id2thread_.at(thrd_loc_id)->GetMsgChannelPtr();
}

void ThreadMgr::InitFromProto(const PbRpf<TaskProto>& tasks) {
  for (auto it = tasks.begin(); it != tasks.end(); ++it) {
    if (it->machine_id() != RuntimeInfo::Singleton().this_machine_id()) {
      continue;
    }
    if (thrd_loc_id2thread_.find(it->thrd_local_id()) == thrd_loc_id2thread_.end()) {
      auto thrd = of_make_unique<Thread>();
      thrd->set_thrd_loc_id(it->thrd_local_id());
      CHECK(thrd_loc_id2thread_.emplace(it->thrd_local_id(), std::move(thrd)).second);
    }
    auto& thrd = thrd_loc_id2thread_.at(it->thrd_local_id());
    thrd->AddActor(*it);
  }
}

void ThreadMgr::Join() {
  for (auto& pair : thrd_loc_id2thread_) {
    pair.second->Join();
  }
}

ThreadMgr::~ThreadMgr() {
  Join();
  thrd_loc_id2thread_.clear();
}

}  // namespace oneflow
