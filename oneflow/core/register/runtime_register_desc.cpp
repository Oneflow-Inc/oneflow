#include "oneflow/core/register/runtime_register_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

RtRegstDesc::RtRegstDesc(const RegstDescProto& regst_desc_proto) {
  regst_desc_id_ = regst_desc_proto.regst_desc_id();
  producer_actor_id_ = regst_desc_proto.producer_task_id();
  consumers_actor_id_ = PbRf2StdVec(regst_desc_proto.consumer_task_id());
  register_num_ = regst_desc_proto.register_num();
  mem_case_ = regst_desc_proto.mem_case();
  for (const auto& pair : regst_desc_proto.lbn2blob_desc()) {
    auto blob_desc = of_make_unique<BlobDesc>(pair.second);
    CHECK(lbn2blob_desc_.emplace(pair.first, std::move(blob_desc)).second);
  }
  auto it = lbn2blob_desc_.begin();
  packed_blob_desc_ = ComputePackedBlobDesc([&]() {
    const BlobDesc* ret = nullptr;
    if (it != lbn2blob_desc_.end()) {
      ret = it->second.get();
      ++it;
    }
    return ret;
  });
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
