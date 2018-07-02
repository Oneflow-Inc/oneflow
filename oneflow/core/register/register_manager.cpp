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
  std::vector<const RegstDescProto*> sorted_regst_descs;
  sorted_regst_descs.reserve(regst_protos.size());
  HashMap<MemoryCase, char*> mem_case2mem_ptr;
  HashMap<MemoryCase, size_t> mem_case2mem_size;
  HashMap<std::pair<MemoryCase, int32_t>, size_t> mem_case7mem_shared_id2size;
  for (const RegstDescProto* regst_desc : regst_protos) {
    int32_t mem_shared_id = regst_desc->mem_sharing_info().mem_shared_id();
    auto rt_regst_desc_ptr = std::make_unique<RtRegstDesc>(*regst_desc);
    if (mem_shared_id != -1) {
      CHECK_EQ(regst_desc->register_num(), 1);
      for (const auto& pair : rt_regst_desc_ptr->GetSize4AllActuallyMemCase()) {
        auto mem_region = std::make_pair(pair.first, mem_shared_id);
        mem_case7mem_shared_id2size[mem_region] =
            std::max(mem_case7mem_shared_id2size[mem_region], pair.second);
      }
    } else {
      for (const auto& pair : rt_regst_desc_ptr->GetSize4AllActuallyMemCase()) {
        mem_case7mem_shared_id2size[std::make_pair(pair.first, mem_shared_id)] +=
            pair.second * regst_desc->register_num();
      }
    }
    sorted_regst_descs.push_back(regst_desc);
    CHECK(regst_desc_id2rt_regst_desc_
              .emplace(regst_desc->regst_desc_id(), std::move(rt_regst_desc_ptr))
              .second);
  }
  for (const auto& pair : mem_case7mem_shared_id2size) {
    mem_case2mem_size[pair.first.first] += pair.second;
  }
  for (const auto& pair : mem_case2mem_size) {
    CHECK(
        mem_case2mem_ptr
            .emplace(pair.first, Global<MemoryAllocator>::Get()->Allocate(pair.first, pair.second))
            .second);
  }
  std::sort(sorted_regst_descs.begin(), sorted_regst_descs.end(),
            [](const RegstDescProto* lhs, const RegstDescProto* rhs) {
              int32_t lhs_mem_shared_id = lhs->mem_sharing_info().mem_shared_id();
              int32_t rhs_mem_shared_id = rhs->mem_sharing_info().mem_shared_id();
              int32_t lhs_used_order_value = lhs->mem_sharing_info().used_order_value();
              int32_t rhs_used_order_value = rhs->mem_sharing_info().used_order_value();
              if (lhs_mem_shared_id == rhs_mem_shared_id) {
                if (lhs_mem_shared_id != -1) {
                  CHECK_NE(lhs_used_order_value, rhs_used_order_value);
                  return lhs_used_order_value < rhs_used_order_value;
                }
                return lhs->regst_desc_id() < rhs->regst_desc_id();
              }
              return lhs_mem_shared_id < rhs_mem_shared_id;
            });
  for (const RegstDescProto* regst_desc : sorted_regst_descs) {
    regst_desc_id2rt_regst_desc_.at(regst_desc->regst_desc_id())->PickMemory(mem_case2mem_ptr);
  }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto, DeviceType device_type,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst(rt_regst_desc);
    if (regst_desc_type.has_normal_regst_desc()) {
      rt_regst_desc->AllocMem4Regst(regst, i, device_type);
    } else if (regst_desc_type.has_record_regst_desc()) {
      const RecordTypeProto& record_type = regst_desc_type.record_regst_desc().record_type();
      switch (record_type) {
        case kOFRecord: regst->packed_blob_.reset(new RecordBlob<OFRecord>); break;
        default: UNIMPLEMENTED();
      }
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

}  // namespace oneflow
