#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

RegstMgr::RegstMgr(const Plan& plan) {
  std::list<const RegstDescProto*> regst_protos;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    for (const auto& pair : task.produced_regst_desc()) { regst_protos.push_back(&pair.second); }
  }
  InitFromRegstProtoList(regst_protos);
}

RegstMgr::RegstMgr(const std::list<const RegstDescProto*>& regst_protos) {
  InitFromRegstProtoList(regst_protos);
}

void RegstMgr::InitFromRegstProtoList(const std::list<const RegstDescProto*>& regst_protos) {
  HashMap<std::pair<MemoryCase, int32_t>, size_t> mem_case7mem_shared_id2mem_size;
  HashMap<std::pair<MemoryCase, int32_t>, char*> mem_case7mem_shared_id2mem_ptr;
  for (const RegstDescProto* regst_desc : regst_protos) {
    auto rt_regst_desc_ptr = std::make_unique<RtRegstDesc>(*regst_desc);
    int32_t mem_shared_id = regst_desc->mem_shared_id();
    for (const auto& pair : rt_regst_desc_ptr->GetSize4AllActuallyMemCase()) {
      size_t regst_mem_case_total_size = pair.second * regst_desc->register_num();
      if (mem_shared_id == -1) {
        char* mem_ptr =
            Global<MemoryAllocator>::Get()->Allocate(pair.first, regst_mem_case_total_size);
        rt_regst_desc_ptr->PickMemory(pair.first, mem_ptr);
      } else {
        auto mem_region = std::make_pair(pair.first, mem_shared_id);
        mem_case7mem_shared_id2mem_size[mem_region] =
            std::max(mem_case7mem_shared_id2mem_size[mem_region], regst_mem_case_total_size);
      }
    }
    CHECK(regst_desc_id2rt_regst_desc_
              .emplace(regst_desc->regst_desc_id(), std::move(rt_regst_desc_ptr))
              .second);
  }
  for (const auto& pair : mem_case7mem_shared_id2mem_size) {
    CHECK(mem_case7mem_shared_id2mem_ptr
              .emplace(pair.first,
                       Global<MemoryAllocator>::Get()->Allocate(pair.first.first, pair.second))
              .second);
  }
  for (const RegstDescProto* regst_desc : regst_protos) {
    int32_t mem_shared_id = regst_desc->mem_shared_id();
    if (mem_shared_id != -1) {
      int64_t regst_desc_id = regst_desc->regst_desc_id();
      auto& rt_regst_desc_ptr = regst_desc_id2rt_regst_desc_.at(regst_desc_id);
      for (const auto& pair : rt_regst_desc_ptr->GetSize4AllActuallyMemCase()) {
        auto mem_region = std::make_pair(pair.first, mem_shared_id);
        char* mem_ptr = mem_case7mem_shared_id2mem_ptr.at(mem_region);
        rt_regst_desc_ptr->PickMemory(pair.first, mem_ptr);
      }
    }
  }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto, DeviceType device_type,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst(rt_regst_desc);
    if (regst_desc_type.has_data_regst_desc()) {
      regst->InitBlobs(rt_regst_desc->GetMemPtrOfMemCase4Regst(i), device_type);
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

}  // namespace oneflow
