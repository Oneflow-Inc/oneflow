#ifndef ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_
#define ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_

#include "oneflow/core/register/register_warpper.h"
#include "oneflow/core/job/keyword.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

class RemoteRegstWarpper final : public RegstWarpper {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RemoteRegstWarpper)
  RemoteRegstWarpper() = delete;
  ~RemoteRegstWarpper() = default;

  RemoteRegstWarpper(Regst* regst) {
    regst_ = regst;

    Blob* blob = regst_->GetBlobPtrFromLbn(kBaledBlobName);
    baled_blob_shape_.reset(new Shape(blob->shape()));
    baled_blob_.reset(new Blob(blob->mut_dptr(), baled_blob_shape_.get()));

    piece_id_ = regst_->piece_id();
    model_version_id_ = regst_->model_version_id();
    regst_desc_id_ = regst_->regst_desc_id();
    producer_actor_id_ = regst_->producer_actor_id();
  }

  Blob* GetBlobPtrFromLbn(const std::string& lbn) override {
    CHECK_EQ(kBaledBlobName, lbn);
    return baled_blob_.get();
  }
  int64_t piece_id() const override {
    return piece_id_;
  }
  int64_t model_version_id() const override {
    return model_version_id_;
  }
  int64_t regst_desc_id() const override {
    return regst_desc_id_;
  }
  int64_t producer_actor_id() const override {
    return producer_actor_id_;
  }
  Regst* regst_raw_ptr() const override {
    return regst_;
  }

 private:
  int64_t piece_id_;
  int64_t model_version_id_;
  int64_t regst_desc_id_;
  int64_t producer_actor_id_;
  Regst* regst_;
  std::unique_ptr<Shape> baled_blob_shape_;
  std::unique_ptr<Blob> baled_blob_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_
