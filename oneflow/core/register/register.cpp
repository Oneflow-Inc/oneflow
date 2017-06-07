#include "oneflow/core/register/register.h"

namespace oneflow {

Regst::Regst() {
  piece_id_ = std::numeric_limits<uint64_t>::max();
  model_version_id_ = std::numeric_limits<uint64_t>::max();
  regst_id_ = std::numeric_limits<uint64_t>::max();
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

} // namespace oneflow
