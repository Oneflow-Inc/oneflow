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
  if (piece_id() == rhs->piece_id()) {
    CHECK_EQ(max_col_id(), rhs->max_col_id());
    return col_id() == rhs->col_id() + 1;
  } else {
    return false;
  }
}

bool Regst::HaveSamePieceColStatusAs(const Regst* rhs) const {
  if (piece_id() == rhs->piece_id()) {
    CHECK_EQ(max_col_id(), rhs->max_col_id());
    return col_id() == rhs->col_id();
  } else {
    return false;
  }
}

Blob* Regst::GetBlobByLbn(const std::string& lbn) {
  auto it = lbn2blob_.find(lbn);
  if (it != lbn2blob_.end()) {
    return static_cast<Blob*>(it->second.get());
  } else if (lbn == kPackedBlobName) {
    return static_cast<Blob*>(packed_blob_.get());
  } else {
    return nullptr;
  }
}

}  // namespace oneflow
