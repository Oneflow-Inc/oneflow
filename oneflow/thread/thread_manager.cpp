#include "thread/thread_manager.h"
#include "thread/comm_bus.h"

namespace oneflow{

// TODO(liuguo): get machine_id of curr machine
extern uint64_t this_machine_id;

void ThreadMgr::InitFromProto(const PbRpf<TaskProto>& tasks) {
  HashMap<uint64_t, Thread*> loc_id2thrd;
  for (auto it = tasks.begin(); it != tasks.end(); ++it) {
    if (it->machine_id() != this_machine_id) { continue; }
    if (loc_id2thrd.find(it->thrd_local_id()) == loc_id2thrd.end()) {
      Thread* thrd = new Thread;
      thrd->set_thrd_loc_id(it->thrd_local_id());
      CHECK(loc_id2thrd.emplace(it->thrd_local_id(), thrd).second);
      CommBus::Singleton().InsertThrdLocIdMsgQPair(
          it->thrd_local_id(), &(thrd->GetMsgQueue()));
      // TODO(liuguo): some other operation on thread if needed
    }
    Thread* thrd = loc_id2thrd.at(it->thrd_local_id());
    thrd->AddActor(*it);
  }
}

}  // namespace oneflow
