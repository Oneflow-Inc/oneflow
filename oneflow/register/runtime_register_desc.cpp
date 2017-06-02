#include "register/runtime_register_desc.h"
#include "common/id_manager.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& regst_desc_proto) {
  regst_desc_id_ = regst_desc_proto.regst_desc_id();
  producer_actor_id_ = 
    IDMgr::Singleton().GetActorIdFromTaskId(regst_desc_proto.producer_task_id());
  register_num_ = regst_desc_proto.register_num();

  const auto& subscriber = regst_desc_proto.subscriber_task_id();
  subscribers_actor_id_.reserve(subscriber.size());
  for (uint64_t task_id : subscriber) {
    subscribers_actor_id_.push_back(IDMgr::Singleton().GetActorIdFromTaskId(task_id));
  }

  for (const auto& pair : regst_desc_proto.lbn2shape()) {
    lbn2shape_.emplace(pair.first, of_make_unique<Shape>(pair.second));
  }
  mem_case_ = regst_desc_proto.mem_case();
}

} // namespace oneflow
