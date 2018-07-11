#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/keyword.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& proto) {
  regst_desc_id_ = proto.regst_desc_id();
  producer_actor_id_ = proto.producer_task_id();
  consumers_actor_id_ = PbRf2StdVec(proto.consumer_task_id());
  register_num_ = proto.register_num();
  mem_case_ = proto.mem_case();
  mem_shared_id_ = proto.mem_shared_id();
  if (proto.regst_desc_type().has_data_regst_desc()) {
    const DataRegstDesc& data_regst_desc = proto.regst_desc_type().data_regst_desc();
    sorted_lbis_.reserve(data_regst_desc.lbi2blob_desc_size());
    for (const LbiBlobDescPair& pair : data_regst_desc.lbi2blob_desc()) {
      sorted_lbis_.push_back(pair.lbi());
      auto blob_desc = std::make_unique<BlobDesc>(pair.blob_desc());
      AccumulateActuallyMemCaseSize(blob_desc.get());
      CHECK(lbi2blob_desc_.emplace(pair.lbi(), std::move(blob_desc)).second);
    }
    CHECK(!sorted_lbis_.empty());
    std::sort(sorted_lbis_.begin(), sorted_lbis_.end());
    packed_blob_desc_ = BlobDesc(data_regst_desc.packed_blob_desc());
  }
}

const BlobDesc* RtRegstDesc::GetBlobDescFromLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2blob_desc_.find(lbi);
  if (it == lbi2blob_desc_.end()) {
    CHECK(lbi.is_packed_id());
    return &packed_blob_desc_;
  } else {
    return it->second.get();
  }
}

void RtRegstDesc::AccumulateActuallyMemCaseSize(const BlobDesc* blob_desc) {
  if (mem_case_.has_host_mem() && mem_case_.host_mem().used_by_network()) {
    mem_case2mem_size_[mem_case_] += blob_desc->TotalByteSize();
  } else {
    MemoryCase header_mem_case;
    header_mem_case.mutable_host_mem();
    mem_case2mem_size_[header_mem_case] += blob_desc->ByteSizeOfHeaderField();
    mem_case2mem_size_[mem_case_] += blob_desc->AlignSizeOfDataContentField();
  }
}

void RtRegstDesc::PickMemory(const MemoryCase& mem_case, char* mem_ptr) {
  auto mem_ptr_it = mem_case2mem_ptr_.find(mem_case);
  CHECK(mem_ptr_it == mem_case2mem_ptr_.end());
  mem_case2mem_ptr_[mem_case] = mem_ptr;
}

void RtRegstDesc::PickOrOccupyMemoryFromMemBlock(HashMap<MemoryCase, char*>& mem_case2mem_ptr,
                                                 bool occupy) {
  auto GetOrOccupyMemPtr = [&](const MemoryCase& mem_case, bool occupy) {
    auto mem_case2mem_ptr_it = mem_case2mem_ptr.find(mem_case);
    CHECK(mem_case2mem_ptr_it != mem_case2mem_ptr.end());
    char* ret = mem_case2mem_ptr_it->second;
    if (occupy) {
      mem_case2mem_ptr.at(mem_case) += mem_case2mem_size_.at(mem_case) * register_num_;
    }
    return ret;
  };
  for (const auto& pair : mem_case2mem_size_) {
    mem_case2mem_ptr_[pair.first] = GetOrOccupyMemPtr(pair.first, occupy);
  }
}

HashMap<MemoryCase, char*> RtRegstDesc::GetMemPtrOfMemCase4Regst(int64_t regst_index) const {
  HashMap<MemoryCase, char*> ret;
  for (const auto& pair : mem_case2mem_ptr_) {
    ret[pair.first] = pair.second + regst_index * mem_case2mem_size_.at(pair.first);
  }
  return ret;
}

}  // namespace oneflow
