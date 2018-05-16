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
  for (const LbiBlobDescPair& pair : proto.lbi2blob_desc()) {
    auto blob_desc = of_make_unique<BlobDesc>(pair.blob_desc());
    CHECK(lbi2blob_desc_.emplace(pair.lbi(), std::move(blob_desc)).second);
  }
  packed_blob_desc_ = BlobDesc(proto.packed_blob_desc());
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

}  // namespace oneflow
