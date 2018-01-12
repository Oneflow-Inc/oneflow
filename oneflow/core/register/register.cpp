#include "oneflow/core/register/register.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

const std::vector<int64_t>& Regst::consumers_actor_id() const {
  return regst_desc_->consumers_actor_id();
}

Regst::Regst() {
  piece_id_ = -1;
  model_version_id_ = -1;
  recurrent_flag_ = 0;
  is_forward_ = true;
  regst_desc_ = nullptr;
}

bool Regst::HaveNextPieceColStatusOf(const Regst* other) const {
  return (piece_id() == other->piece_id())
         && (max_col_id() == other->max_col_id())
         && (col_id() == other->col_id() + 1);
}

Blob* Regst::GetBlobByLbn(const std::string& lbn) {
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
