#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/protobuf.h"
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
      auto blob_desc = std::make_unique<RtBlobDesc>(pair.blob_desc());
      CHECK(lbi2blob_desc_.emplace(pair.lbi(), std::move(blob_desc)).second);
    }
    packed_blob_desc_.reset(new RtBlobDesc(data_regst_desc.packed_blob_desc()));
    CHECK(data_regst_desc.has_time_shape());
    data_regst_time_shape_.reset(new Shape(data_regst_desc.time_shape()));
  } else {
    packed_blob_desc_.reset(new RtBlobDesc(BlobDesc(DataType::kChar)));
  }
}

const RtBlobDesc* RtRegstDesc::GetRtBlobDescFromLbi(const LogicalBlobId& lbi) const {
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
  return MainByteSize4OneRegst() * register_num_;
}

size_t RtRegstDesc::MainByteSize4OneRegst() const {
  if (packed_blob_desc_->is_body_disabled()) {
    if (mem_case_.has_device_cuda_mem()) {
      return 0;
    } else {
      return packed_blob_desc_->ByteSizeOfBlobHeader();
    }
  } else {
    if (mem_case_.has_device_cuda_mem()) {
      return packed_blob_desc_->ByteSizeOfBlobBody();
    } else {
      return packed_blob_desc_->TotalByteSize();
    }
  }
}

size_t RtRegstDesc::TotalSeparatedHeaderByteSize4AllRegst() const {
  return SeparatedHeaderByteSize4OneRegst() * register_num_;
}

size_t RtRegstDesc::SeparatedHeaderByteSize4OneRegst() const {
  if (mem_case_.has_device_cuda_mem()) {
    return packed_blob_desc_->ByteSizeOfBlobHeader();
  } else {
    return 0;
  }
}

const Shape& RtRegstDesc::data_regst_time_shape() const {
  CHECK(regst_desc_type_.has_data_regst_desc());
  CHECK(data_regst_time_shape_);
  return *data_regst_time_shape_;
}

void RtRegstDesc::ForEachBlobDescOffsetInOnRegst(
    const std::vector<LbiBlobDescPair>& lbis,
    const std::function<void(const LbiBlobDescPair&, int64_t body_offset, int64_t header_offset)>&
        Handler) const {
  int64_t cur_body_offset = 0;
  int64_t cur_header_offset = 0;
  for (const LbiBlobDescPair& lbi : lbis) {
    Handler(lbi, cur_body_offset, cur_header_offset);
    const RtBlobDesc* blob_desc = GetRtBlobDescFromLbi(lbi.lbi());
    cur_body_offset += blob_desc->ByteSizeOfBlobBody();
    cur_header_offset += blob_desc->ByteSizeOfBlobHeader();
  }
}

}  // namespace oneflow
