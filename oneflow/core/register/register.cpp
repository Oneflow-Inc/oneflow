#include "oneflow/core/register/register.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

Regst::Regst() {
  piece_id_ = -1;
  model_version_id_ = -1;
}

void Regst::ForEachLbn(std::function<void(const std::string&)> func) {
  for (const auto& pair : lbn2blob_) { func(pair.first); }
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  auto it = lbn2blob_.find(lbn);
  if (it != lbn2blob_.end()) {
    return it->second.get();
  } else if (lbn == kPackedBlobName) {
    return packed_blob_.get();
  } else {
    return nullptr;
  }
}

}  // namespace oneflow
