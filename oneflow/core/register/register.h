#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() {
    deleter_();
  }
  
  Blob* GetBlobPtrFromLbn(const std::string& lbn);

  uint64_t regst_desc_id() const {
    return regst_desc_->regst_desc_id();
  }
  uint64_t producer_actor_id() const {
    return regst_desc_->producer_actor_id();
  }
  const std::vector<uint64_t>& subscribers_actor_id() const {
    return regst_desc_->subscribers_actor_id();
  }
    
 private:
  friend class RegstMgr;
  Regst() = default;

  std::shared_ptr<const RtRegstDesc> regst_desc_;
  uint64_t regst_id_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REGISTER_H_
