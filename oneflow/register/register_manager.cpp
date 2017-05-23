#include "register/register_manager.h"

namespace oneflow {

namespace {


MemoryCase InferMemoryCaseFromTaskType(
    const std::pair<int, bool>& producer_type, 
    const HashSet<std::pair<int, bool>, pair_hash>& subscriber_types,
    uint64_t device_id) {
  MemoryCase mem_case;
  if (producer_type.first == DeviceCompTask) {
    mem_case.mutable_device_cuda_mem()->set_device_id(device_id);
  } else if (producer_type.first == HostCompTask or producer_type.first == BoxingTask) {
    mem_case.mutable_host_pageable_mem();
    if (subscriber_types.find(std::make_pair(CopyCommNetTask, false)) != 
        subscriber_types.end()) {
      mem_case.mutable_host_pinned_mem()->set_need_rdma(true);
    }
    if (subscriber_types.find(std::make_pair(CopyHdTask, true)) != 
        subscriber_types.end()) {
      mem_case.mutable_host_pinned_mem()->set_need_cuda(true);
    }
  } else if (producer_type.first == CopyCommNetTask) {
    mem_case.mutable_host_pinned_mem()->set_need_rdma(true);
    if (subscriber_types.find(std::make_pair(CopyHdTask, true)) != 
        subscriber_types.end()) {
      mem_case.mutable_host_pinned_mem()->set_need_cuda(true);
    }
  } else if (producer_type.first == CopyHdTask) {
    if (producer_type.second) {
      mem_case.mutable_device_cuda_mem()->set_device_id(device_id);
    } else {
      mem_case.mutable_host_pinned_mem()->set_need_cuda(true);
      if (subscriber_types.find(std::make_pair(CopyCommNetTask, false)) != 
          subscriber_types.end()) {
        mem_case.mutable_host_pinned_mem()->set_need_rdma(true);
      }
    }
  }
  return mem_case;
}

}

void RegstMgr::NewRegstFromRegstDesc(
    uint64_t producer_id,
    uint64_t device_id,
    const std::pair<int, bool>& producer_type,
    const RegstDescProto& regstdesc,
    std::size_t sizeof_floating,
    const std::vector<uint64_t>& subscriber_ids,
    const HashSet<std::pair<int, bool>, pair_hash>& subscriber_types) {
  uint64_t regst_desc_id = regstdesc.regst_desc_id();
  for (int64_t i = 0; i < regstdesc.register_num(); ++i) {
    std::unique_ptr<Regst> regst(new Regst());
    regst->id_ = IDMgr::Singleton().NewRegstId(regst_desc_id);
    regst->producer_id_ = producer_id;
    regst->subscriber_ids_ = subscriber_ids;
    std::size_t regst_size = 0;
    for (const auto& mpair : regstdesc.lbn2shape()) {
      Shape shape(mpair.second);
      regst_size += shape.elem_cnt() * sizeof_floating;
    }
    MemoryCase mem_case = InferMemoryCaseFromTaskType(producer_type, 
                                                      subscriber_types, 
                                                      device_id);
    auto mem_info = MemoryAllocator::Singleton().Allocate(mem_case, regst_size);
    regst->deleter_ = mem_info.second;
    char* dptr = mem_info.first;
    for (const auto& mpair : regstdesc.lbn2shape()) {
      Shape shape(mpair.second);
      regst->lbn2blob_.emplace(mpair.first, of_make_unique<Blob>(dptr, shape));
      dptr += shape.elem_cnt() * sizeof_floating;
    }
    regst_id2regst_.emplace(regst->id_, std::move(regst));
  }
}

void RegstMgr::InitFromProto(const OfElf& ofelf) {
  HashSet<uint64_t> regst_desc_idsinmachine;
  /*
  //Init all regst for id, cnt, producer_id, lbn2blob
  HashMap<uint64_t, HashSet<uint64_t>> actor_id2produced_regst_desc_id;
  HashMap<uint64_t, std::vector<uint64_t>> regst_desc_id2subscriber_ids;
  HashMap<uint64_t, HashSet<std::pair<int, bool>, pair_hash>> regst_desc_id2subscriber_types;
  for (const TaskProto& taskproto : ofelf.task()) {
    uint64_t actor_id = IDMgr::Singleton().GetActorIdFromTaskId(taskproto.id());
    for (const RegstDescProto& regstdesc : taskproto.produced_regst_desc()) {
      uint64_t regst_desc_id = regstdesc.regst_desc_id();
      actor_id2produced_regst_desc_id[actor_id].insert(regst_desc_id);
      if (taskproto.machine_id() != RuntimeInfo::Singleton().this_machine_id()) {
        regst_desc_idsinmachine.insert(regst_desc_id);
      }
    }
  }
  for (const TaskProto& taskproto : ofelf.task()) {
    uint64_t actor_id = IDMgr::Singleton().GetActorIdFromTaskId(taskproto.id());
    HashSet<uint64_t> processed_regst_desc_ids;
    for (const ExecNodeProto& execnode: taskproto.exec_sequence().exec_node()) {
      for (const auto& mpair : execnode.bn_in_op2regst_desc_id()) {
        if (actor_id2produced_regst_desc_id.at(actor_id).find(mpair.second) != 
            actor_id2produced_regst_desc_id.at(actor_id).end())  { continue; }
        if (processed_regst_desc_ids.find(mpair.second) != 
            processed_regst_desc_ids.end()) { continue; }
        if (regst_desc_idsinmachine.find(mpair.second) != 
            regst_desc_idsinmachine.end()) {
          regst_desc_id2subscriber_ids[mpair.second].push_back(actor_id);
          regst_desc_id2subscriber_types[mpair.second].insert(
              std::make_pair(taskproto.type(), taskproto.is_h2d()));
          processed_regst_desc_ids.insert(mpair.second);
        }
      }
    }
  }
  std::size_t sizeof_floating;
  if (ofelf.job_desc().floating_point_type() == kFloat) {
    sizeof_floating = sizeof(float);
  } else {
    sizeof_floating = sizeof(double);
  }
  for (const TaskProto& taskproto : ofelf.task()) {
    if (taskproto.machine_id() != RuntimeInfo::Singleton().this_machine_id()) { continue; }
    uint64_t actor_id = IDMgr::Singleton().GetActorIdFromTaskId(taskproto.id());
    uint64_t device_id = IDMgr::Singleton().DevPhyId4ThrdLocId(taskproto.thrd_local_id());
    for (const RegstDescProto& regstdesc : taskproto.produced_regst_desc()) {
      uint64_t regst_desc_id = regstdesc.regst_desc_id();
      NewRegstFromRegstDesc(actor_id,
                            device_id,
                            std::make_pair(taskproto.type(), taskproto.is_h2d()),
                            regstdesc, 
                            sizeof_floating, 
                            regst_desc_id2subscriber_ids.at(regst_desc_id), 
                            regst_desc_id2subscriber_types.at(regst_desc_id));
    }
  }
  }*/
  TODO();
}

}
