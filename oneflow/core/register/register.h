#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() { deleter_(); }

  // Getters
  int recurrent_flag() const { return recurrent_flag_; }
  int64_t piece_id() const { return piece_id_; }
  int64_t model_version_id() const { return model_version_id_; }
  int64_t regst_desc_id() const { return regst_desc_->regst_desc_id(); }
  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;
  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  Blob* GetBlobByLbn(const std::string& lbn);
  Blob* packed_blob() { return packed_blob_.get(); }
  const PieceStatus& piece_status() const {
    return lbn2blob_.begin()->second->piece_status();
  }

  // Setters
  void set_piece_id(int64_t val) { piece_id_ = val; }
  void set_model_version_id(int64_t val) { model_version_id_ = val; }
  void set_recurrent_flag(int val) { recurrent_flag_ = val; }

 private:
  friend class RegstMgr;
  Regst();

  int64_t piece_id_;
  int64_t model_version_id_;

  int recurrent_flag_;
  // 0: no recurrent, 1 recurrent from top, -1 recurrent from bot

  const RtRegstDesc* regst_desc_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
  std::unique_ptr<Blob> packed_blob_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
