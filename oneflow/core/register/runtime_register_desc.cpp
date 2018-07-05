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
  header_mem_case_.mutable_host_mem();
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
  if (mem_case_.has_host_mem() && !mem_case_.host_mem().used_by_device()) {
    actually_mem_case2size_[mem_case_] += blob_desc->TotalByteSize();
  } else {
    actually_mem_case2size_[header_mem_case_] += blob_desc->ByteSizeOfHeaderField();
    actually_mem_case2size_[mem_case_] += blob_desc->AlignSizeOfDataContentField();
  }
}

std::pair<size_t, size_t> RtRegstDesc::SizeOfBlobField(const BlobDesc* blob_desc) const {
  std::pair<size_t, size_t> ret = {0, 0};
  if (mem_case_.has_host_mem() && !mem_case_.host_mem().used_by_device()) {
    ret.first = blob_desc->TotalByteSize();
  } else {
    ret.first = blob_desc->ByteSizeOfHeaderField();
    ret.second = blob_desc->AlignSizeOfDataContentField();
  }
  return ret;
}

void RtRegstDesc::PickMemory(const MemoryCase& mem_case, char* mem_ptr) {
  auto mem_ptr_it = mem_case2mem_ptr_.find(mem_case);
  CHECK(mem_ptr_it == mem_case2mem_ptr_.end());
  mem_case2mem_ptr_[mem_case] = mem_ptr;
}

void RtRegstDesc::PickMemoryFromMemBlock(HashMap<MemoryCase, char*>& mem_case2mem_ptr,
                                         bool need_move_ptr) {
  auto GetMemPtrAndMoveIt = [&](const MemoryCase& mem_case) {
    char* ret = mem_case2mem_ptr.at(mem_case);
    mem_case2mem_ptr.at(mem_case) += actually_mem_case2size_.at(mem_case) * register_num_;
    return ret;
  };
  for (const auto& pair : actually_mem_case2size_) {
    mem_case2mem_ptr_[pair.first] =
        need_move_ptr ? GetMemPtrAndMoveIt(pair.first) : mem_case2mem_ptr.at(pair.first);
  }
}

void RtRegstDesc::AllocMem4Regst(Regst* regst, int64_t index, DeviceType device_type) {
  char* packed_mem_ptr = nullptr;
  char* cur_header_mem_ptr = nullptr;
  char* cur_data_mem_ptr = nullptr;
  size_t cur_header_mem_size = 0;
  size_t cur_data_mem_size = 0;
  if (mem_case_.has_host_mem() && !mem_case_.host_mem().used_by_device()) {
    cur_header_mem_ptr =
        mem_case2mem_ptr_.at(mem_case_) + index * actually_mem_case2size_.at(mem_case_);
    packed_mem_ptr = cur_header_mem_ptr;
  } else {
    cur_header_mem_ptr = mem_case2mem_ptr_.at(header_mem_case_)
                         + index * actually_mem_case2size_.at(header_mem_case_);
    cur_data_mem_ptr =
        mem_case2mem_ptr_.at(mem_case_) + index * actually_mem_case2size_.at(mem_case_);
    packed_mem_ptr = cur_data_mem_ptr;
  }
  for (const LogicalBlobId& lbi : sorted_lbis_) {
    const BlobDesc* blob_desc = GetBlobDescFromLbi(lbi);
    std::tie(cur_header_mem_size, cur_data_mem_size) = SizeOfBlobField(blob_desc);
    Blob* blob = NewBlob(regst, blob_desc, cur_header_mem_size > 0 ? cur_header_mem_ptr : nullptr,
                         cur_data_mem_size > 0 ? cur_data_mem_ptr : nullptr, device_type);
    regst->AddBlob(lbi, blob);
    if (cur_header_mem_size > 0) { cur_header_mem_ptr += cur_header_mem_size; }
    if (cur_data_mem_size > 0) { cur_data_mem_ptr += cur_data_mem_size; }
  }
  Blob* packed_blob = NewBlob(regst, &packed_blob_desc_, packed_mem_ptr, device_type);
  regst->set_packed_blob(packed_blob);
  if (mem_case_.has_host_mem() && mem_case_.host_mem().used_by_network()) {
    regst->set_comm_net_token(
        Global<CommNet>::Get()->RegisterMemory(packed_mem_ptr, packed_blob_desc_.TotalByteSize()));
  }
}

}  // namespace oneflow
