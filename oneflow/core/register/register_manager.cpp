#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace std {

template<>
struct hash<oneflow::MemoryCase> {
  size_t operator()(const oneflow::MemoryCase& val) const {
    if (val.has_host_mem()) {
      return val.host_mem().used_by_device() + 1024;
    } else {
      return val.device_cuda_mem().device_id();
    }
  }
};

}  // namespace std

namespace oneflow {

inline bool operator==(const MemoryCase& lhs, const MemoryCase& rhs) {
  if (lhs.has_host_mem() && rhs.has_host_mem()) {
    return lhs.host_mem().used_by_device() == rhs.host_mem().used_by_device();
  }
  if (lhs.has_device_cuda_mem() && rhs.has_device_cuda_mem()) {
    return lhs.device_cuda_mem().device_id() == rhs.device_cuda_mem().device_id();
  }
  return false;
}

RegstMgr::RegstMgr(const Plan& plan) {
  HashMap<MemoryCase, char*> mem_case2mem_ptr;
  HashMap<MemoryCase, size_t> mem_case2mem_size;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst_desc_proto = pair.second;
      CHECK(regst_desc_id2rt_regst_desc_
                .emplace(regst_desc_proto.regst_desc_id(),
                         std::make_unique<const RtRegstDesc>(regst_desc_proto))
                .second);
      const MemoryCase& mem_case = regst_desc_proto.mem_case();
      if (mem_case2mem_size.find(mem_case) == mem_case2mem_size.end()) {
        mem_case2mem_size.emplace(mem_case, 0);
      }
      mem_case2mem_size[mem_case] += (regst_desc_id2rt_regst_desc_[regst_desc_proto.regst_desc_id()]
                                          ->packed_blob_desc()
                                          ->TotalByteSize()
                                      * regst_desc_proto.register_num());
    }
  }
  for (const auto& pair : mem_case2mem_size) {
    const MemoryCase& mem_case = pair.first;
    std::tuple<char*, std::function<void()>> allocation_result =
        Global<MemoryAllocator>::Get()->Allocate(mem_case, pair.second);
    CHECK(mem_case2mem_ptr.emplace(mem_case, std::get<0>(allocation_result)).second);
    deleters_.push_back(std::get<1>(allocation_result));
  }
  for (const auto& pair : regst_desc_id2rt_regst_desc_) {
    const int64_t& regst_desc_id = pair.first;
    const RtRegstDesc* rt_regst_desc = pair.second.get();
    const MemoryCase& mem_case = rt_regst_desc->mem_case();
    CHECK(regst_desc_id2mem_ptr_.emplace(regst_desc_id, mem_case2mem_ptr[mem_case]).second);
    mem_case2mem_ptr[mem_case] +=
        (rt_regst_desc->packed_blob_desc()->TotalByteSize() * rt_regst_desc->register_num());
  }
}

RegstMgr::~RegstMgr() {
  for (std::function<void()> deleter : deleters_) { deleter(); }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto, DeviceType device_type,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  const RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_[regst_desc_id].get();
  char* mem_ptr = regst_desc_id2mem_ptr_[regst_desc_id];
  std::vector<LogicalBlobId> lbis;
  if (regst_desc_type.has_normal_regst_desc()) {
    for (const LbiBlobDescPair& pair : regst_desc_type.normal_regst_desc().lbi2blob_desc()) {
      lbis.push_back(pair.lbi());
    }
    CHECK(!lbis.empty());
  }
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst;
    regst->regst_desc_ = rt_regst_desc;
    if (regst_desc_type.has_normal_regst_desc()) {
      std::sort(lbis.begin(), lbis.end());
      char* cur_pointer = mem_ptr;
      for (const LogicalBlobId& lbi : lbis) {
        const BlobDesc* blob_desc = rt_regst_desc->GetBlobDescFromLbi(lbi);
        std::unique_ptr<Blob> blob_ptr;
        blob_ptr.reset(NewBlob(regst, blob_desc, cur_pointer, device_type));
        CHECK(regst->lbi2blob_.emplace(lbi, std::move(blob_ptr)).second);
        cur_pointer += blob_desc->TotalByteSize();
      }
      regst->packed_blob_.reset(
          NewBlob(regst, rt_regst_desc->packed_blob_desc(), mem_ptr, device_type));
      if (rt_regst_desc->mem_case().has_host_mem()
          && rt_regst_desc->mem_case().host_mem().used_by_network()) {
        regst->comm_net_token_ = Global<CommNet>::Get()->RegisterMemory(
            mem_ptr, rt_regst_desc->packed_blob_desc()->TotalByteSize());
      }
      mem_ptr += rt_regst_desc->packed_blob_desc()->TotalByteSize();
    } else if (regst_desc_type.has_record_regst_desc()) {
      const RecordTypeProto& record_type = regst_desc_type.record_regst_desc().record_type();
      switch (record_type) {
        case kOFRecord: regst->packed_blob_.reset(new RecordBlob<OFRecord>); break;
        default: UNIMPLEMENTED();
      }
    } else if (regst_desc_type.has_delay_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

}  // namespace oneflow
