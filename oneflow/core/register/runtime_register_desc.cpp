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
  for (const auto& pair : proto.lbn2blob_desc()) {
    auto blob_desc = of_make_unique<BlobDesc>(pair.second);
    CHECK(lbn2blob_desc_.emplace(pair.first, std::move(blob_desc)).second);
  }
  packed_blob_desc_ = BlobDesc(proto.packed_blob_desc());
}

const BlobDesc* RtRegstDesc::GetBlobDescFromLbn(const std::string& lbn) const {
  auto it = lbn2blob_desc_.find(lbn);
  if (it == lbn2blob_desc_.end()) {
    CHECK_STREQ(lbn.c_str(), kPackedBlobName);
    return &packed_blob_desc_;
  } else {
    return it->second.get();
  }
}

}  // namespace oneflow
