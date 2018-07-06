#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

struct RegstStatus {
  int64_t regst_desc_id;
  int64_t piece_id;
  int64_t model_version_id;
  int64_t act_id;
  int32_t col_id;
  int32_t max_col_id;
};

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst();

  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;

  const HashMap<LogicalBlobId, std::unique_ptr<Blob>>& lbi2blob() const { return lbi2blob_; }
  Blob* packed_blob() const { return static_cast<Blob*>(packed_blob_.get()); }
  void set_packed_blob(Blob* packed_blob) { packed_blob_.reset(packed_blob); }
  Blob* GetBlobByLbi(const LogicalBlobId& lbi) const;
  void AddBlob(LogicalBlobId lbi, Blob* blob);

  void* comm_net_token() const { return comm_net_token_; }
  void set_comm_net_token(void* comm_net_token) { comm_net_token_ = comm_net_token; }

  const RegstStatus& status() const { return status_; }
  int64_t piece_id() const { return status_.piece_id; }
  int64_t model_version_id() const { return status_.model_version_id; }
  int64_t act_id() const { return status_.act_id; }
  int32_t col_id() const { return status_.col_id; }
  int32_t max_col_id() const { return status_.max_col_id; }
  bool IsMaxCol() const { return col_id() == max_col_id(); }
  int64_t regst_desc_id() const {
    CHECK_NE(status_.regst_desc_id, -1);
    return status_.regst_desc_id;
  }

  void set_piece_id(int64_t val) { status_.piece_id = val; }
  void set_model_version_id(int64_t val) { status_.model_version_id = val; }
  void set_act_id(int64_t val) { status_.act_id = val; }
  void set_col_id(int32_t val) { status_.col_id = val; }
  void set_max_col_id(int32_t val) { status_.max_col_id = val; }

 private:
  friend class RegstMgr;
  Regst() = delete;
  Regst(const RtRegstDesc*);

  const RtRegstDesc* regst_desc_;
  HashMap<LogicalBlobId, std::unique_ptr<Blob>> lbi2blob_;
  std::unique_ptr<Blob> packed_blob_;
  void* comm_net_token_;
  RegstStatus status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
