#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

class PieceStatus final {
 public:
  PieceStatus() : piece_id_(0), col_id_(0), max_col_id_(-1) {}
  ~PieceStatus() = default;
  PieceStatus(const PieceStatus&) = default;
  PieceStatus& operator=(const PieceStatus&) = default;

  bool operator==(const PieceStatus& other) const {
    return (piece_id_ == other.piece_id_) && (col_id_ == other.col_id_)
           && (max_col_id_ == other.max_col_id_);
  }
  bool operator!=(const PieceStatus& other) const { return !(*this == other); }

  int64_t piece_id() const { return piece_id_; }
  int64_t col_id() const { return col_id_; }
  int64_t max_col_id() const { return max_col_id_; }

  void set_max_col_id(int64_t max_col_id) {
    CHECK_EQ(-1, max_col_id_);  //-1 for unset
    max_col_id_ = max_col_id;
  }

  int GetIntoNextStatus();
  bool IsLast() const;
  bool IsLastCol() const { return col_id_ == max_col_id_; }
  bool IsNextColOf(const PieceStatus& pre) const;

 private:
  int64_t piece_id_;
  int64_t col_id_;
  int64_t max_col_id_;
};

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() { deleter_(); }

  // Getters
  const PieceStatus& piece_status() const { return piece_status_; }
  int recurrent_flag() const { return recurrent_flag_; }
  bool is_forward() const { return is_forward_; }
  int64_t piece_id() const { return piece_id_; }
  int64_t model_version_id() const { return model_version_id_; }
  int64_t regst_desc_id() const { return regst_desc_->regst_desc_id(); }
  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;
  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  Blob* GetBlobByLbn(const std::string& lbn);
  Blob* packed_blob() { return packed_blob_.get(); }

  // Setters
  void set_piece_id(int64_t val) { piece_id_ = val; }
  void set_model_version_id(int64_t val) { model_version_id_ = val; }
  void set_piece_status(const PieceStatus& pst) { piece_status_ = pst; }
  void set_recurrent_flag(int val) { recurrent_flag_ = val; }
  void set_is_forward(bool val) { is_forward_ = true; }

 private:
  friend class RegstMgr;
  Regst();

  int64_t piece_id_;
  int64_t model_version_id_;

  PieceStatus piece_status_;
  int recurrent_flag_;
  // 0: no recurrent, 1 recurrent from top, -1 recurrent from bot
  bool is_forward_;  // true for fw regst, false for bp regst

  const RtRegstDesc* regst_desc_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
  std::unique_ptr<Blob> packed_blob_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
