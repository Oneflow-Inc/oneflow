#include "oneflow/core/register/register.h"

namespace oneflow {

Regst::Regst() {
  piece_id_ = std::numeric_limits<uint64_t>::max();
  model_version_id_ = std::numeric_limits<uint64_t>::max();
  regst_id_ = std::numeric_limits<uint64_t>::max();
}

void Regst::ForEachLbn(std::function<void(const std::string&)> func) {
  for (const auto& pair : lbn2blob_) {
    func(pair.first);
  }
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

} // namespace oneflow
