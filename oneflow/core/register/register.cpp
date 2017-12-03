#include "oneflow/core/register/register.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

int PieceStatus::GetIntoNextStatus() {
  if (IsLast()) { return -1; }
  if (col_id_ == max_col_id_) {
    piece_id_ += 1;
    col_id_ = 0;
    max_col_id_ = -1;
  } else {
    col_id_ += 1;
  }
  return 0;
}

bool PieceStatus::IsLast() const {
  if (piece_id_ == JobDesc::Singleton()->total_piece_num() - 1
      && col_id_ == max_col_id_) {
    return true;
  }
  return false;
}

bool PieceStatus::IsNextColOf(const PieceStatus& pre) const {
  if (piece_id_ == pre.piece_id_ && max_col_id_ == pre.max_col_id_
      && col_id_ == pre.col_id_ + 1) {
    return true;
  }
  return false;
}

Regst::Regst() {
  piece_id_ = -1;
  model_version_id_ = -1;
  recurrent_flag_ = 0;
  is_forward_ = true;
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
