#include "thread/thread_manager.h"
#include "thread/actor_msg_bus.h"
#include "job/runtime_info.h"

namespace oneflow{

void ThreadMgr::InitFromProto(const PbRpf<TaskProto>& tasks) {
  HashMap<uint64_t, Thread*> loc_id2thrd;
  for (auto it = tasks.begin(); it != tasks.end(); ++it) {
    if (it->machine_id() != RuntimeInfo::Singleton().machine_id()) {
      continue;
    }
    if (loc_id2thrd.find(it->thrd_local_id()) == loc_id2thrd.end()) {
      auto thrd = of_make_unique<Thread>();
      thrd->set_thrd_loc_id(it->thrd_local_id());
      CHECK(loc_id2thrd.emplace(it->thrd_local_id(), std::move(thrd)).second);
      ActorMsgBus::Singleton().InsertThrdLocIdMsgQPair(
          it->thrd_local_id(), &(thrd->GetMsgQueue()));
      // TODO(liuguo): some other operation on thread if needed
    }
    Thread* thrd = loc_id2thrd.at(it->thrd_local_id());
    thrd->AddActor(*it);
  }
}

}  // namespace oneflow
