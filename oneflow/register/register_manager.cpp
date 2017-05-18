#include "register/register_manager.h"

namespace oneflow {

void RegstMgr::InitFromProto(const OfElf& ofelf) {
  //Init all regst for id, cnt, producer_id, lbn2blob
  HashMap<uint64_t, std::unordered_set<uint64_t>> actor_id2produced_regst_desc_id;
  HashMap<uint64_t, std::vector<uint64_t>> regst_desc_id2regst_ids;
  for (const TaskProto& taskproto : ofelf.task()) {
    uint64_t actor_id = IDMgr::Singleton().TaskId2ActorId(taskproto.id());
    for (const RegstDescProto& regstdesc : taskproto.produced_regst_desc()) {
      uint64_t regst_desc_id = regstdesc.regst_desc_id();
      for (int64_t i = 0; i < regstdesc.register_num(); ++i) {
        auto regst = of_make_unique<Regst>();
        regst->id_ = IDMgr::Singleton().NewRegstId(regst_desc_id);
        regst->cnt_.store(0);
        regst->producer_id_ = actor_id;
        for (const auto& mpair : regstdesc.lbn2shape()) {
          regst->lbn2blob_.emplace(mpair.first, new Blob());
        }
        regst_id2regst_.emplace(regst->id_, std::move(regst));
        actor_id2produced_regst_desc_id[actor_id].insert(regst_desc_id);
        regst_desc_id2regst_ids[regst_desc_id].push_back(regst->id_);
      }
    }
  }
  //for consumer_ids, lbn2blob
  for (const TaskProto& taskproto : ofelf.task()) {
    uint64_t actor_id = IDMgr::Singleton().TaskId2ActorId(taskproto.id());
    std::unordered_set<uint64_t> processed_consumer;
    for (const ExecNodeProto& execnode: taskproto.exec_sequence().exec_node()) {
      for (const auto& mpair : execnode.bn_in_op2regst_desc_id()) {
        if (actor_id2produced_regst_desc_id.at(actor_id).find(mpair.second) != 
            actor_id2produced_regst_desc_id.at(actor_id).end())  { continue; }
        if (processed_consumer.find(mpair.second) != processed_consumer.end()) { continue; }
        for (uint64_t regst_id : regst_desc_id2regst_ids[mpair.second]) {
          GetRegstFromRegstID(regst_id)->consumer_ids_.push_back(actor_id);
        }
      }
    }
  }
}

}
