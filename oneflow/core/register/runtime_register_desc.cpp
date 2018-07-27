#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& proto) {
  regst_desc_id_ = proto.regst_desc_id();
  producer_actor_id_ = proto.producer_task_id();
  consumers_actor_id_ = PbRf2StdVec(proto.consumer_task_id());
  register_num_ = proto.register_num();
  mem_case_ = proto.mem_case();
  regst_desc_type_ = proto.regst_desc_type();
  if (proto.regst_desc_type().has_data_regst_desc()) {
    const DataRegstDesc& data_regst_desc = proto.regst_desc_type().data_regst_desc();
    for (const LbiBlobDescPair& pair : data_regst_desc.lbi2blob_desc()) {
      auto blob_desc = std::make_unique<BlobDesc>(pair.blob_desc());
      CHECK(lbi2blob_desc_.emplace(pair.lbi(), std::move(blob_desc)).second);
    }
    if (data_regst_desc.packed_blob_desc().has_header_byte_size_for_mem_blob()) {
      packed_blob_desc_.reset(new MemBlobDesc(data_regst_desc.packed_blob_desc()));
    } else {
      packed_blob_desc_.reset(new BlobDesc(data_regst_desc.packed_blob_desc()));
    }
  }
}

const BlobDesc* RtRegstDesc::GetBlobDescFromLbi(const LogicalBlobId& lbi) const {
  auto it = lbi2blob_desc_.find(lbi);
  if (it == lbi2blob_desc_.end()) {
    CHECK(lbi.is_packed_id());
    return packed_blob_desc_.get();
  } else {
    return it->second.get();
  }
}

size_t RtRegstDesc::TotalByteSize4AllRegst() const {
  return packed_blob_desc_->TotalByteSize() * register_num_;
}

size_t RtRegstDesc::TotalMainByteSize4AllRegst() const {
  if (mem_case_.has_device_cuda_mem()) {
    return packed_blob_desc_->ByteSizeOfDataContentField() * register_num_;
  } else {
    return packed_blob_desc_->TotalByteSize() * register_num_;
  }
}

size_t RtRegstDesc::MainByteSize4OneRegst() const {
  if (mem_case_.has_device_cuda_mem()) {
    return packed_blob_desc_->ByteSizeOfDataContentField();
  } else {
    return packed_blob_desc_->TotalByteSize();
  }
}

size_t RtRegstDesc::TotalSeparatedByteSize4AllRegst() const {
  if (mem_case_.has_device_cuda_mem()) {
    return packed_blob_desc_->ByteSizeOfBlobHeader() * register_num_;
  } else {
    return 0;
  }
}

}  // namespace oneflow
