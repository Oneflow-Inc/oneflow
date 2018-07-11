#include "oneflow/core/register/register.h"
#include "oneflow/core/job/keyword.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

Regst::Regst(const RtRegstDesc* regst_desc) : regst_desc_(regst_desc), comm_net_token_(nullptr) {
  status_.regst_desc_id = regst_desc->regst_desc_id();
  status_.piece_id = -1;
  status_.model_version_id = -1;
  status_.act_id = -1;
  status_.col_id = 0;
  status_.max_col_id = 0;
}

Regst::~Regst() {
  if (comm_net_token_ != nullptr) { Global<CommNet>::Get()->UnRegisterMemory(comm_net_token_); }
}

const std::vector<int64_t>& Regst::consumers_actor_id() const {
  return regst_desc_->consumers_actor_id();
}

Blob* Regst::GetBlobByLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2blob_.find(lbi);
  if (it != lbi2blob_.end()) {
    return static_cast<Blob*>(it->second.get());
  } else if (lbi.is_packed_id()) {
    return static_cast<Blob*>(packed_blob_.get());
  } else {
    return nullptr;
  }
}

void Regst::InitBlobs(HashMap<MemoryCase, char*> mem_case2mem_ptr, DeviceType device_type) {
  const MemoryCase& mem_case = regst_desc_->mem_case();
  auto GetOrOccupyMemPtr = [&](const BlobDesc* blob_desc, bool occupy) -> std::pair<char*, char*> {
    std::pair<char*, char*> ret = {nullptr, nullptr};
    if (mem_case.has_host_mem() && mem_case.host_mem().used_by_network()) {
      size_t total_mem_size = blob_desc->TotalByteSize();
      auto mem_case_ptr_it = mem_case2mem_ptr.find(mem_case);
      CHECK(mem_case_ptr_it != mem_case2mem_ptr.end());
      ret.first = mem_case_ptr_it->second;
      if (occupy) { mem_case2mem_ptr.at(mem_case) += total_mem_size; }
    } else {
      size_t header_mem_size = blob_desc->ByteSizeOfHeaderField();
      if (header_mem_size > 0) {
        MemoryCase header_mem_case;
        header_mem_case.mutable_host_mem();
        auto header_mem_case_ptr_it = mem_case2mem_ptr.find(header_mem_case);
        CHECK(header_mem_case_ptr_it != mem_case2mem_ptr.end());
        ret.first = header_mem_case_ptr_it->second;
        if (occupy) { mem_case2mem_ptr.at(header_mem_case) += header_mem_size; }
      }
      size_t data_mem_size = blob_desc->AlignSizeOfDataContentField();
      if (data_mem_size > 0) {
        auto mem_case_ptr_it = mem_case2mem_ptr.find(mem_case);
        CHECK(mem_case_ptr_it != mem_case2mem_ptr.end());
        ret.second = mem_case_ptr_it->second;
        if (occupy) { mem_case2mem_ptr.at(mem_case) += data_mem_size; }
      }
    }
    return ret;
  };

  const BlobDesc* packed_blob_desc = regst_desc_->packed_blob_desc();
  auto packed_mem_ptr = GetOrOccupyMemPtr(packed_blob_desc, false);
  packed_blob_.reset(
      NewBlob(this, packed_blob_desc, packed_mem_ptr.first, packed_mem_ptr.second, device_type));
  if (mem_case.has_host_mem() && mem_case.host_mem().used_by_network()) {
    comm_net_token_ = Global<CommNet>::Get()->RegisterMemory(packed_mem_ptr.first,
                                                             packed_blob_desc->TotalByteSize());
  }

  for (const LogicalBlobId& lbi : regst_desc_->sorted_lbis()) {
    const BlobDesc* blob_desc = regst_desc_->GetBlobDescFromLbi(lbi);
    auto mem_ptr_pair = GetOrOccupyMemPtr(blob_desc, true);
    Blob* blob = NewBlob(this, blob_desc, mem_ptr_pair.first, mem_ptr_pair.second, device_type);
    blob->InitOFRecordBlobIfNeed();
    AddBlob(lbi, blob);
  }
}

void Regst::AddBlob(LogicalBlobId lbi, Blob* blob) {
  std::unique_ptr<Blob> blob_ptr(blob);
  CHECK(lbi2blob_.emplace(lbi, std::move(blob_ptr)).second);
}

}  // namespace oneflow
