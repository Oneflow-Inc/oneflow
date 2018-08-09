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
  HashMap<MemoryCase, char*> mem_case2mem_ptr;
  HashMap<MemoryCase, size_t> mem_case2mem_size;
  std::vector<const RegstDescProto*> sorted_regst_protos(regst_protos.begin(), regst_protos.end());
  for (const RegstDescProto* regst_desc : regst_protos) {
    CHECK(
        regst_desc_id2rt_regst_desc_
            .emplace(regst_desc->regst_desc_id(), std::make_unique<const RtRegstDesc>(*regst_desc))
            .second);
    if (regst_desc->mem_shared_id() != -1) { CHECK_EQ(regst_desc->register_num(), 1); }
  }
  auto GetRegstSize = [&](const RegstDescProto* regst_desc) {
    return regst_desc_id2rt_regst_desc_.at(regst_desc->regst_desc_id())->TotalByteSize4AllRegst();
  };
  std::sort(sorted_regst_protos.begin(), sorted_regst_protos.end(),
            [&](const RegstDescProto* lhs, const RegstDescProto* rhs) {
              return (lhs->mem_shared_id() < rhs->mem_shared_id())
                     || (lhs->mem_shared_id() == rhs->mem_shared_id()
                         && GetRegstSize(lhs) > GetRegstSize(rhs));
            });
  int32_t last_mem_shared_id = -1;
  char* mem_ptr = nullptr;
  for (const RegstDescProto* regst_desc : sorted_regst_protos) {
    if (regst_desc->regst_desc_type().has_data_regst_desc() == false) { continue; }
    CHECK_GT(GetRegstSize(regst_desc), 0);
    int32_t current_mem_shared_id = regst_desc->mem_shared_id();
    if (current_mem_shared_id == -1 || (current_mem_shared_id != last_mem_shared_id)) {
      mem_ptr = Global<MemoryAllocator>::Get()->Allocate(regst_desc->mem_case(),
                                                         GetRegstSize(regst_desc));
    }
    CHECK(regst_desc_id2mem_ptr_.emplace(regst_desc->regst_desc_id(), mem_ptr).second);
    last_mem_shared_id = current_mem_shared_id;
  }
}

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
                         std::function<void(Regst*)> OneRegstDone) {
  const int64_t regst_desc_id = regst_desc_proto.regst_desc_id();
  const RegstDescTypeProto& regst_desc_type = regst_desc_proto.regst_desc_type();
  const RtRegstDesc* rt_regst_desc = regst_desc_id2rt_regst_desc_.at(regst_desc_id).get();
  char* mem_ptr = nullptr;
  if (regst_desc_id2mem_ptr_.find(regst_desc_id) != regst_desc_id2mem_ptr_.end()) {
    mem_ptr = regst_desc_id2mem_ptr_.at(regst_desc_id);
  }
  std::vector<LogicalBlobId> lbis;
  if (regst_desc_type.has_data_regst_desc()) {
    for (const LbiBlobDescPair& pair : regst_desc_type.data_regst_desc().lbi2blob_desc()) {
      lbis.push_back(pair.lbi());
    }
    CHECK(!lbis.empty());
    CHECK(mem_ptr != nullptr);
  }
  for (int64_t i = 0; i < rt_regst_desc->register_num(); ++i) {
    Regst* regst = new Regst;
    regst->set_regst_desc(rt_regst_desc);
    if (regst_desc_type.has_data_regst_desc()) {
      std::sort(lbis.begin(), lbis.end());
      char* cur_pointer = mem_ptr;
      for (const LogicalBlobId& lbi : lbis) {
        const BlobDesc* blob_desc = rt_regst_desc->GetBlobDescFromLbi(lbi);
        std::unique_ptr<Blob> blob_ptr(new Blob(regst, blob_desc, cur_pointer));
        InitPbBlobIfNeed(blob_ptr.get());
        CHECK(regst->lbi2blob_.emplace(lbi, std::move(blob_ptr)).second);
        cur_pointer += blob_desc->TotalByteSize();
      }
      regst->packed_blob_.reset(new Blob(regst, rt_regst_desc->packed_blob_desc(), mem_ptr));
      if (rt_regst_desc->mem_case().has_host_mem()
          && rt_regst_desc->mem_case().host_mem().used_by_network()) {
        regst->comm_net_token_ = Global<CommNet>::Get()->RegisterMemory(
            mem_ptr, rt_regst_desc->packed_blob_desc()->TotalByteSize());
      }
      mem_ptr += rt_regst_desc->packed_blob_desc()->TotalByteSize();
    } else if (regst_desc_type.has_ctrl_regst_desc()) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    OneRegstDone(regst);
  }
}

void RegstMgr::InitPbBlobIfNeed(Blob* blob_ptr) const {
  switch (blob_ptr->blob_desc().data_type()) {
#define INIT_PB_BLOB_CASE(type_cpp, type_proto) \
  case type_proto: return InitPbBlob<type_cpp>(blob_ptr);
    OF_PP_FOR_EACH_TUPLE(INIT_PB_BLOB_CASE, RECORD_DATA_TYPE_SEQ);

    default: CHECK_NE(blob_ptr->blob_desc().data_type(), DataType::kInvalidDataType);
  }
}

template<typename T>
void RegstMgr::InitPbBlob(Blob* blob_ptr) const {
  FOR_RANGE(int64_t, idx, 0, blob_ptr->blob_desc().shape().elem_cnt()) {
    Global<MemoryAllocator>::Get()->PlacementNew(&blob_ptr->mut_dptr<T>()[idx]);
  }
}

}  // namespace oneflow
