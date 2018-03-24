#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
                         DeviceType device_type, RecordTypeProto record_type,
                         std::function<void(Regst*)> OneRegstDone) {
  const RtRegstDesc* runtime_regst_desc = new RtRegstDesc(regst_desc_proto);
  rt_regst_descs_.emplace_back(runtime_regst_desc);
  for (int64_t i = 0; i < regst_desc_proto.register_num(); ++i) {
    Regst* regst = new Regst;
    regst->regst_desc_ = runtime_regst_desc;
    std::vector<std::string> lbns;
    for (const auto& pair : regst_desc_proto.lbn2blob_desc()) {
      lbns.push_back(pair.first);
    }
    if (lbns.size() > 0) {
      std::sort(lbns.begin(), lbns.end());
      std::tuple<char*, const void*, std::function<void()>> allocation_result =
          Global<MemoryAllocator>::Get()->Allocate(
              regst_desc_proto.mem_case(),
              runtime_regst_desc->packed_blob_desc()->TotalByteSize());
      char* cur_pointer = std::get<0>(allocation_result);
      for (const std::string& lbn : lbns) {
        const BlobDesc* blob_desc = runtime_regst_desc->GetBlobDescFromLbn(lbn);
        std::unique_ptr<Blob> blob_ptr;
        blob_ptr.reset(
            NewBlob(regst, blob_desc, cur_pointer, nullptr, device_type));
        CHECK(regst->lbn2blob_.emplace(lbn, std::move(blob_ptr)).second);
        cur_pointer += blob_desc->TotalByteSize();
      }
      regst->packed_blob_.reset(
          NewBlob(regst, runtime_regst_desc->packed_blob_desc(),
                  std::get<0>(allocation_result),
                  std::get<1>(allocation_result), device_type));
      regst->deleter_ = std::get<2>(allocation_result);
    } else {
      switch (record_type) {
        case kOFRecord:
          regst->packed_blob_.reset(new RecordBlob<OFRecord>);
          break;
      }
      regst->deleter_ = []() {};
    }
    OneRegstDone(regst);
  }
}

}  // namespace oneflow
