#ifndef ONEFLOW_CORE_REGISTER_LOCAL_REGISTER_WRAPPER_H_
#define ONEFLOW_CORE_REGISTER_LOCAL_REGISTER_WRAPPER_H_

#include "oneflow/core/register/register_wrapper.h"

namespace oneflow {

class LocalRegstWrapper final : public RegstWrapper {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalRegstWrapper);
  LocalRegstWrapper() = delete;
  ~LocalRegstWrapper() = default;

  LocalRegstWrapper(Regst* regst) : regst_(regst) {}

  Blob* GetBlobPtrFromLbn(const std::string& lbn) override {
    return regst_->GetBlobPtrFromLbn(lbn);
  }
  int64_t piece_id() const override { return regst_->piece_id(); }
  int64_t model_version_id() const override {
    return regst_->model_version_id();
  }
  int64_t regst_desc_id() const override { return regst_->regst_desc_id(); }
  int64_t producer_actor_id() const override {
    return regst_->producer_actor_id();
  }
  Regst* regst_raw_ptr() const override { return regst_; }

 private:
  Regst* regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_LOCAL_REGISTER_WRAPPER_H_
