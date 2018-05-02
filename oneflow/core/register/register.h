#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

struct RegstStatus {
  int64_t piece_id;
  int64_t model_version_id;
  int64_t act_id;
  int32_t col_id;
  int32_t max_col_id;
};

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() { deleter_(); }

  // Getters
  const RegstStatus& status() const { return status_; }
  int64_t piece_id() const { return status_.piece_id; }
  int64_t model_version_id() const { return status_.model_version_id; }
  int64_t act_id() const { return status_.act_id; }
  int32_t col_id() const { return status_.col_id; }
  int32_t max_col_id() const { return status_.max_col_id; }

  int64_t regst_desc_id() const { return regst_desc_->regst_desc_id(); }
  int64_t producer_actor_id() const { return regst_desc_->producer_actor_id(); }
  const std::vector<int64_t>& consumers_actor_id() const;
  const RtRegstDesc* regst_desc() const { return regst_desc_; }
  Blob* GetBlobByLbi(const LogicalBlobId& lbi);
  const HashMap<LogicalBlobId, std::unique_ptr<BlobIf>>& lbi2blob() const { return lbi2blob_; }

  Blob* packed_blob() { return static_cast<Blob*>(packed_blob_.get()); }
  template<typename RecordType>
  RecordBlob<RecordType>* GetRecordBlob() {
    return static_cast<RecordBlob<RecordType>*>(packed_blob_.get());
  }
  RecordBlobIf* GetRecordBlobIf() { return static_cast<RecordBlobIf*>(packed_blob_.get()); }

  bool IsMaxCol() const { return col_id() == max_col_id(); }

  // Setters
  void set_piece_id(int64_t val) { status_.piece_id = val; }
  void set_model_version_id(int64_t val) { status_.model_version_id = val; }
  void set_act_id(int64_t val) { status_.act_id = val; }
  void set_col_id(int32_t val) { status_.col_id = val; }
  void set_max_col_id(int32_t val) { status_.max_col_id = val; }

 private:
  friend class RegstMgr;
  Regst();

  RegstStatus status_;
  const RtRegstDesc* regst_desc_;
  std::function<void()> deleter_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobIf>> lbi2blob_;
  std::unique_ptr<BlobIf> packed_blob_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_H_
