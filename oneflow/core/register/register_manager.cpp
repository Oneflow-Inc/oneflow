#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

namespace {

int64_t MemCase2DevMemId(const MemoryCase& mem_case) {
  if (mem_case.has_host_mem()) {
    if (mem_case.host_mem().used_by_device()) {
      return Global<JobDesc>::Get()->resource().gpu_device_num() + 1;
    } else {
      return Global<JobDesc>::Get()->resource().gpu_device_num();
    }
  } else if (mem_case.has_device_cuda_mem()) {
    return mem_case.device_cuda_mem().device_id();
  } else {
    UNIMPLEMENTED();
  }
}

MemoryCase DevMemId2MemCase(const int64_t dev_mem_id) {
  int64_t gpu_device_num = Global<JobDesc>::Get()->resource().gpu_device_num();
  MemoryCase ret;
  if (dev_mem_id < gpu_device_num) {
    ret.mutable_device_cuda_mem()->set_device_id(dev_mem_id);
  } else if (dev_mem_id == gpu_device_num) {
    ret.mutable_host_mem();
  } else if (dev_mem_id == gpu_device_num + 1) {
    ret.mutable_host_mem()->set_used_by_device(true);
  } else {
    UNIMPLEMENTED();
  }
  return ret;
}

}  // namespace

RegstMgr::RegstMgr(const Plan& plan) {
  int64_t gpu_device_num = Global<JobDesc>::Get()->resource().gpu_device_num();
  int64_t dev_mem_num = gpu_device_num + 2;
  // device mem id:
  // 0 ~ gpu_device_num - 1        each gpu device mem
  // gpu_device_num                CPU normal mem
  // gpu_device_num + 1            CPU mem used by device
  std::vector<int64_t> dev_mem_id2mem_size(dev_mem_num, 0);
  std::vector<int64_t> dev_mem_id2mem_offset(dev_mem_num, 0);
  std::vector<char*> dev_mem_id2mem_ptr(dev_mem_num, nullptr);
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != Global<MachineCtx>::Get()->this_machine_id()) { continue; }
    for (const auto& pair : task.produced_regst_desc()) {
      const RegstDescProto& regst_desc_proto = pair.second;
      CHECK(regst_desc_id2rt_regst_desc_
                .emplace(regst_desc_proto.regst_desc_id(),
                         std::unique_ptr<const RtRegstDesc>(new RtRegstDesc(regst_desc_proto)))
                .second);
      dev_mem_id2mem_size[MemCase2DevMemId(regst_desc_proto.mem_case())] +=
          (regst_desc_id2rt_regst_desc_[regst_desc_proto.regst_desc_id()]
               ->packed_blob_desc()
               ->TotalByteSize()
           * regst_desc_proto.register_num());
    }
  }
  for (size_t i = 0; i < dev_mem_num; ++i) {
    if (i < gpu_device_num) { CudaCheck(cudaSetDevice(i)); }
    int32_t current_device_id;
    CudaCheck(cudaGetDevice(&current_device_id));
    std::tuple<char*, std::function<void()>> allocation_result =
        Global<MemoryAllocator>::Get()->Allocate(DevMemId2MemCase(i), dev_mem_id2mem_size[i]);
    dev_mem_id2mem_ptr[i] = std::get<0>(allocation_result);
    CHECK(dev_id2deleters_.emplace(i, std::get<1>(allocation_result)).second);
  }
  for (const auto& pair : regst_desc_id2rt_regst_desc_) {
    const int64_t& regst_desc_id = pair.first;
    const RtRegstDesc* rt_regst_desc = pair.second.get();
    const int64_t dev_mem_id = MemCase2DevMemId(rt_regst_desc->mem_case());
    CHECK(regst_desc_id2mem_ptr_
              .emplace(regst_desc_id,
                       dev_mem_id2mem_ptr[dev_mem_id] + dev_mem_id2mem_offset[dev_mem_id])
              .second);
    dev_mem_id2mem_offset[dev_mem_id] +=
        (rt_regst_desc->packed_blob_desc()->TotalByteSize() * rt_regst_desc->register_num());
    CHECK_LE(dev_mem_id2mem_offset[dev_mem_id], dev_mem_id2mem_size[dev_mem_id]);
  }
  for (size_t i = 0; i < dev_mem_num; ++i) {
    CHECK_EQ(dev_mem_id2mem_offset[i], dev_mem_id2mem_size[i]);
  }
}

RegstMgr::~RegstMgr() {
  for (void* comm_net_token : comm_net_tokens_) {
    Global<CommNet>::Get()->UnRegisterMemory(comm_net_token);
  }
  for (const auto& pair : dev_id2deleters_) {
    if (pair.first < Global<JobDesc>::Get()->resource().gpu_device_num()) {
      CudaCheck(cudaSetDevice(pair.first));
    }
    pair.second();
  }
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
        void* comm_net_token = Global<CommNet>::Get()->RegisterMemory(
            mem_ptr, rt_regst_desc->packed_blob_desc()->TotalByteSize());
        regst->comm_net_token_ = comm_net_token;
        comm_net_tokens_.push_back(comm_net_token);
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
