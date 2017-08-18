#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& regst_desc_proto) {
  regst_desc_id_ = regst_desc_proto.regst_desc_id();
  producer_actor_id_ =
      IDMgr::Singleton()->ActorId4TaskId(regst_desc_proto.producer_task_id());
  register_num_ = regst_desc_proto.register_num();

  const auto& consumers = regst_desc_proto.consumer_task_id();
  consumers_actor_id_.reserve(consumers.size());
  for (int64_t task_id : consumers) {
    consumers_actor_id_.push_back(IDMgr::Singleton()->ActorId4TaskId(task_id));
  }

  for (const auto& pair : regst_desc_proto.lbn2blob_desc()) {
    CHECK(lbn2blob_desc_
              .emplace(pair.first, of_make_unique<BlobDesc>(pair.second))
              .second);
  }
  mem_case_ = regst_desc_proto.mem_case();
}

}  // namespace oneflow
