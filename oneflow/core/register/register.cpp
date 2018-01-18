#include "oneflow/core/register/register.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

const std::vector<int64_t>& Regst::consumers_actor_id() const {
  return regst_desc_->consumers_actor_id();
}

Regst::Regst() {
  status_.piece_id = -1;
  status_.model_version_id = -1;
  status_.act_id = -1;
  status_.col_id = 0;
  status_.max_col_id = 0;
  regst_desc_ = nullptr;
}

bool Regst::HaveNextPieceColStatusOf(const Regst* rhs) const {
  return (piece_id() == rhs->piece_id()) && (max_col_id() == rhs->max_col_id())
         && (col_id() == rhs->col_id() + 1);
}

bool Regst::HaveSamePieceColStatusAs(const Regst* rhs) const {
  return (piece_id() == rhs->piece_id()) && (max_col_id() == rhs->max_col_id())
         && (col_id() == rhs->col_id());
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
