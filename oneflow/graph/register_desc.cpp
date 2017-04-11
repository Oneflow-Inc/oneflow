#include "graph/register_desc.h"
#include "job/id_manager.h"

namespace oneflow {

RegstDesc::RegstDesc() {
  regst_desc_id_ = IDMgr::Singleton().NewRegstDescId();
  producer_ = nullptr;
}

void RegstDesc::EnrollWithPbnAndLbn(const std::string& pbn,
                                    const std::string& lbn) {
  std::unique_ptr<BlobDesc> blob_desc(new BlobDesc);
  blob_desc->mut_pbn() = pbn;
  blob_desc->mut_lbn() = lbn;
  CHECK(pbn2blob_desc_.insert(
        std::make_pair(pbn, std::move(blob_desc))).second);
}

void RegstDesc::EnrollWithLbn(const std::string& lbn) {
  std::unique_ptr<BlobDesc> blob_desc(new BlobDesc);
  blob_desc->mut_pbn() =
      "regst_desc_" + std::to_string(regst_desc_id_) + "/" + lbn;
  blob_desc->mut_lbn() = lbn;
  CHECK(lbn2blob_desc_.insert(
        std::make_pair(lbn, std::move(blob_desc))).second);
}

const char* ContigRegstDesc::kAllLbn = "OfReservedAllLbn";

Shape ContigRegstDesc::GetLbnShape(const std::string& lbn) const {
  if (lbn == kAllLbn) {
    return ComputeShape4AllLbn();
  } else {
    return RegstDesc::GetLbnShape(lbn);
  }
}

Shape ContigRegstDesc::ComputeShape4AllLbn() const {
  TODO();
}

} // namespace oneflow
