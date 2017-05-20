#include "register/register_manager.h"

namespace oneflow {

void RegstMgr::NewRegstFromRegstDesc(
    uint64_t producer_id,
    const RegstDescProto& regstdesc,
    std::size_t sizeof_floating,
    HashMap<uint64_t, HashSet<uint64_t>>& actor_id2produced_regst_desc_id,
    HashMap<uint64_t, std::vector<uint64_t>>& regst_desc_id2regst_ids) {
  uint64_t regst_desc_id = regstdesc.regst_desc_id();
  for (int64_t i = 0; i < regstdesc.register_num(); ++i) {
    std::unique_ptr<Regst> regst(new Regst());
    regst->id_ = IDMgr::Singleton().NewRegstId(regst_desc_id);
    regst->producer_id_ = producer_id;
    std::size_t regst_size = 0;
    for (const auto& mpair : regstdesc.lbn2shape()) {
      Shape shape(mpair.second);
      regst_size += shape.elem_cnt() * sizeof_floating;
    }
    auto mem_info = MemoryAllocator::Singleton().Allocate(MemoryCase(),
                                                          regst_size);
    regst->deleter_ = mem_info.second;
    char* dptr = mem_info.first;
    for (const auto& mpair : regstdesc.lbn2shape()) {
      Shape shape(mpair.second);
      regst->lbn2blob_.emplace(mpair.first, of_make_unique<Blob>(dptr, shape));
      dptr += shape.elem_cnt() * sizeof_floating;
    }
    regst_id2regst_.emplace(regst->id_, std::move(regst));
    actor_id2produced_regst_desc_id[producer_id].insert(regst_desc_id);
    regst_desc_id2regst_ids[regst_desc_id].push_back(regst->id_);
  }
}

void RegstMgr::InitFromProto(const OfElf& ofelf) {
  //Init all regst for id, cnt, producer_id, lbn2blob
  HashMap<uint64_t, HashSet<uint64_t>> actor_id2produced_regst_desc_id;
  HashMap<uint64_t, std::vector<uint64_t>> regst_desc_id2regst_ids;
  std::size_t sizeof_floating;
  if (ofelf.job_desc().floating_point_type() == kFloat) {
    sizeof_floating = sizeof(float);
  } else {
    sizeof_floating = sizeof(double);
  }
  for (const TaskProto& taskproto : ofelf.task()) {
    if (taskproto.machine_id() != RuntimeInfo::Singleton().this_machine_id()) { continue; }
    uint64_t actor_id = IDMgr::Singleton().GetActorIdFromTaskId(taskproto.id());
    for (const RegstDescProto& regstdesc : taskproto.produced_regst_desc()) {
      NewRegstFromRegstDesc(actor_id, 
                            regstdesc, 
                            sizeof_floating, 
                            actor_id2produced_regst_desc_id, 
                            regst_desc_id2regst_ids);
    }
  }
  //for consumer_ids, lbn2blob
  for (const TaskProto& taskproto : ofelf.task()) {
    if (taskproto.machine_id() != RuntimeInfo::Singleton().this_machine_id()) { continue; }
    uint64_t actor_id = IDMgr::Singleton().GetActorIdFromTaskId(taskproto.id());
    HashSet<uint64_t> processed_consumer;
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
