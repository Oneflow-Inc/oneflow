#ifndef ONEFLOW_CORE_REGISTER_REGISTER_WARPPER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_WARPPER_H_

#include "oneflow/core/register/register.h"

namespace oneflow {

class RegstWarpper {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(RegstWarpper);
  virtual ~RegstWarpper() = default;

  virtual Blob* GetBlobPtrFromLbn(const std::string& lbn) = 0;
  virtual uint64_t piece_id() const = 0;
  virtual uint64_t model_version_id() const = 0;
  virtual uint64_t regst_desc_id() const = 0;
  virtual uint64_t producer_actor_id() const = 0;
  virtual Regst* regst_raw_ptr() const = 0;

 protected:
  RegstWarpper() = default;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REGISTER_WARPPER_H_
