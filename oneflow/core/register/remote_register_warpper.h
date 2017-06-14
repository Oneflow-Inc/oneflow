#ifndef ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_
#define ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_

#include "oneflow/core/register/register_warpper.h"

namespace oneflow {

class RemoteRegstWarpper final : public RegstWarpper {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(RemoteRegstWarpper)
  RemoteRegstWarpper() = delete;
  ~RemoteRegstWarpper() = default;

  RemoteRegstWarpper(Regst* regst) {
    TODO();
  }

  Blob* GetBlobPtrFromLbn(const std::string& lbn) override { TODO(); }
  uint64_t piece_id() const override { TODO(); }
  uint64_t model_version_id() const override { TODO(); }
  uint64_t regst_desc_id() const override { TODO(); }
  uint64_t producer_actor_id() const override { TODO(); }
  Regst* regst_raw_ptr() const override { TODO(); }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REMOTE_REGISTER_WARPPER_H_
